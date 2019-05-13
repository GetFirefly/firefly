#![allow(unused)]
//! A contiguous growable array type with heap-allocated contents, same
//! as the standard library `Vec<T>`, but also parameterized on the backing
//! allocator
use core::cmp::Ordering;
use core::fmt;
use core::hash::{self, Hash};
use core::intrinsics::{arith_offset, assume};
use core::iter::{FusedIterator, TrustedLen};
use core::mem;
use core::ops::{self, Index, IndexMut};
use core::ptr::{self, NonNull};
use core::slice::{self, SliceIndex};
use core::alloc::Alloc;
use core::marker::PhantomData;

use crate::alloc::alloc_ref::{AllocRef, AsAllocRef};
use crate::alloc::raw_vec::RawVec;
use crate::alloc::boxed::Box;

pub struct Vec<'a, T, A: AllocRef<'a>> {
    buf: RawVec<'a, T, A>,
    len: usize,
}


impl<'a, T, A: ?Sized + Alloc + Sync, H: AllocRef<'a, Alloc=A>> Vec<'a, T, H> {
    #[inline]
    pub fn new<R: 'a + AsAllocRef<'a, Handle=H>>(a: &'a R) -> Self {
        Vec {
            buf: RawVec::new_in(a),
            len: 0,
        }
    }

    #[inline]
    pub fn with_capacity<R: 'a + AsAllocRef<'a, Handle=H>>(capacity: usize, a: &'a R) -> Self {
        Vec {
            buf: RawVec::with_capacity_in(capacity, a),
            len: 0,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.cap()
    }

    pub fn reserve(&mut self, additional: usize) {
        self.buf.reserve(self.len, additional);
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        self.buf.reserve_exact(self.len, additional);
    }

    pub fn shrink_to_fit(&mut self) {
        if self.capacity() != self.len {
            self.buf.shrink_to_fit(self.len);
        }
    }

    pub fn into_boxed_slice(mut self) -> Box<'a, [T], H> {
        unsafe {
            self.shrink_to_fit();
            let buf = ptr::read(&self.buf);
            mem::forget(self);
            buf.into_box()
        }
    }

    pub fn truncate(&mut self, len: usize) {
        let current_len = self.len;
        unsafe {
            let mut ptr = self.as_mut_ptr().add(self.len);
            // Set the final length at the end, keeping in mind that
            // dropping an element might panic. Works around a missed
            // optimization, as seen in the following issue:
            // https://github.com/rust-lang/rust/issues/51802
            let mut local_len = SetLenOnDrop::new(&mut self.len);

            // drop any extra elements
            for _ in len..current_len {
                local_len.decrement_len(1);
                ptr = ptr.offset(-1);
                ptr::drop_in_place(ptr);
            }
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity());

        self.len = new_len;
    }

    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        unsafe {
            // We replace self[index] with the last element. Note that if the
            // bounds check on hole succeeds there must be a last element (which
            // can be self[index] itself).
            let hole: *mut T = &mut self[index];
            let last = ptr::read(self.get_unchecked(self.len - 1));
            self.len -= 1;
            ptr::replace(hole, last)
        }
    }

    pub fn insert(&mut self, index: usize, element: T) {
        let len = self.len();
        assert!(index <= len);

        // space for the new element
        if len == self.buf.cap() {
            self.reserve(1);
        }

        unsafe {
            // infallible
            // The spot to put the new value
            {
                let p = self.as_mut_ptr().add(index);
                // Shift everything over to make space. (Duplicating the
                // `index`th element into two consecutive places.)
                ptr::copy(p, p.offset(1), len - index);
                // Write it in, overwriting the first copy of the `index`th
                // element.
                ptr::write(p, element);
            }
            self.set_len(len + 1);
        }
    }

    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(index < len);
        unsafe {
            // infallible
            let ret;
            {
                // the place we are taking from.
                let ptr = self.as_mut_ptr().add(index);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                ret = ptr::read(ptr);

                // Shift everything down to fill in that spot.
                ptr::copy(ptr.offset(1), ptr, len - index - 1);
            }
            self.set_len(len - 1);
            ret
        }
    }

    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F) where F: FnMut(&mut T) -> K, K: PartialEq {
        self.dedup_by(|a, b| key(a) == key(b))
    }

    pub fn dedup_by<F>(&mut self, same_bucket: F) where F: FnMut(&mut T, &mut T) -> bool {
        let len = {
            let (dedup, _) = self.as_mut_slice().partition_dedup_by(same_bucket);
            dedup.len()
        };
        self.truncate(len);
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        // This will panic or abort if we would allocate > isize::MAX bytes
        // or if the length increment would overflow for zero-sized types.
        if self.len == self.buf.cap() {
            self.reserve(1);
        }
        unsafe {
            let end = self.as_mut_ptr().add(self.len);
            ptr::write(end, value);
            self.len += 1;
        }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                self.len -= 1;
                Some(ptr::read(self.get_unchecked(self.len())))
            }
        }
    }

    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        unsafe {
            self.append_elements(other.as_slice() as _);
            other.set_len(0);
        }
    }

    #[inline]
    unsafe fn append_elements(&mut self, other: *const [T]) {
        let count = (*other).len();
        self.reserve(count);
        let len = self.len();
        ptr::copy_nonoverlapping(other as *const T, self.get_unchecked_mut(len), count);
        self.len += count;
    }

    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.len(), "`at` out of bounds");

        let other_len = self.len - at;
        let alloc_ref = self.buf.alloc_ref().clone();
        let mut other = Vec::with_capacity_and_alloc_ref(other_len, alloc_ref);

        // Unsafely `set_len` and copy items to `other`.
        unsafe {
            self.set_len(at);
            other.set_len(other_len);

            ptr::copy_nonoverlapping(self.as_ptr().add(at),
                                     other.as_mut_ptr(),
                                     other.len());
        }
        other
    }

    pub fn resize_with<F>(&mut self, new_len: usize, f: F)
        where F: FnMut() -> T
    {
        let len = self.len();
        if new_len > len {
            self.extend_with(new_len - len, ExtendFunc(f));
        } else {
            self.truncate(new_len);
        }
    }
}

impl<'a, T, A: AllocRef<'a>> Vec<'a, T, A> {
    #[inline]
    pub fn with_capacity_and_alloc_ref(capacity: usize, a: A) -> Self {
        Vec {
            buf: RawVec::with_capacity_and_alloc_ref(capacity, a),
            len: 0,
        }
    }
}

impl<'a, T: Clone, A: AllocRef<'a>> Vec<'a, T, A> {
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len();

        if new_len > len {
            self.extend_with(new_len - len, ExtendElement(value))
        } else {
            self.truncate(new_len);
        }
    }

    pub fn extend_from_slice<'b: 'a>(&mut self, other: &'b [T]) {
        self.spec_extend(other.iter())
    }
}

// This code generalises `extend_with_{element,default}`.
trait ExtendWith<T> {
    fn next(&mut self) -> T;
    fn last(self) -> T;
}

struct ExtendElement<T>(T);
impl<T: Clone> ExtendWith<T> for ExtendElement<T> {
    fn next(&mut self) -> T { self.0.clone() }
    fn last(self) -> T { self.0 }
}

struct ExtendDefault;
impl<T: Default> ExtendWith<T> for ExtendDefault {
    fn next(&mut self) -> T { Default::default() }
    fn last(self) -> T { Default::default() }
}

struct ExtendFunc<F>(F);
impl<T, F: FnMut() -> T> ExtendWith<T> for ExtendFunc<F> {
    fn next(&mut self) -> T { (self.0)() }
    fn last(mut self) -> T { (self.0)() }
}

impl<'a, T, A: AllocRef<'a>> Vec<'a, T, A> {
    /// Extend the vector by `n` values, using the given generator.
    fn extend_with<E: ExtendWith<T>>(&mut self, n: usize, mut value: E) {
        self.reserve(n);

        unsafe {
            let mut ptr = self.as_mut_ptr().add(self.len());
            // Use SetLenOnDrop to work around bug where compiler
            // may not realize the store through `ptr` through self.set_len()
            // don't alias.
            let mut local_len = SetLenOnDrop::new(&mut self.len);

            // Write all elements except the last one
            for _ in 1..n {
                ptr::write(ptr, value.next());
                ptr = ptr.offset(1);
                // Increment the length in every step in case next() panics
                local_len.increment_len(1);
            }

            if n > 0 {
                // We can write the last element directly without cloning needlessly
                ptr::write(ptr, value.last());
                local_len.increment_len(1);
            }

            // len set by scope guard
        }
    }

    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize, cap: usize, a: A) -> Self {
        let buf = RawVec::from_raw_parts_and_alloc_ref(ptr, cap, a);
        Self {
            buf,
            len,
        }
    }
}

// Set the length of the vec when the `SetLenOnDrop` value goes out of scope.
//
// The idea is: The length field in SetLenOnDrop is a local variable
// that the optimizer will see does not alias with any stores through the Vec's data
// pointer. This is a workaround for alias analysis issue #32155
struct SetLenOnDrop<'a> {
    len: &'a mut usize,
    local_len: usize,
}

impl<'a> SetLenOnDrop<'a> {
    #[inline]
    fn new(len: &'a mut usize) -> Self {
        SetLenOnDrop { local_len: *len, len: len }
    }

    #[inline]
    fn increment_len(&mut self, increment: usize) {
        self.local_len += increment;
    }

    #[inline]
    fn decrement_len(&mut self, decrement: usize) {
        self.local_len -= decrement;
    }
}

impl Drop for SetLenOnDrop<'_> {
    #[inline]
    fn drop(&mut self) {
        *self.len = self.local_len;
    }
}

impl<'a, T: PartialEq, A: AllocRef<'a>> Vec<'a, T, A> {
    #[inline]
    pub fn dedup(&mut self) {
        self.dedup_by(|a, b| a == b)
    }

    pub fn remove_item(&mut self, item: &T) -> Option<T> {
        let pos = self.iter().position(|x| *x == *item)?;
        Some(self.remove(pos))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Internal methods and functions
////////////////////////////////////////////////////////////////////////////////

unsafe trait IsZero {
    /// Whether this value is zero
    fn is_zero(&self) -> bool;
}

macro_rules! impl_is_zero {
    ($t: ty, $is_zero: expr) => {
        unsafe impl IsZero for $t {
            #[inline]
            fn is_zero(&self) -> bool {
                $is_zero(*self)
            }
        }
    }
}

impl_is_zero!(i8, |x| x == 0);
impl_is_zero!(i16, |x| x == 0);
impl_is_zero!(i32, |x| x == 0);
impl_is_zero!(i64, |x| x == 0);
impl_is_zero!(i128, |x| x == 0);
impl_is_zero!(isize, |x| x == 0);

impl_is_zero!(u16, |x| x == 0);
impl_is_zero!(u32, |x| x == 0);
impl_is_zero!(u64, |x| x == 0);
impl_is_zero!(u128, |x| x == 0);
impl_is_zero!(usize, |x| x == 0);

impl_is_zero!(bool, |x| x == false);
impl_is_zero!(char, |x| x == '\0');

impl_is_zero!(f32, |x: f32| x.to_bits() == 0);
impl_is_zero!(f64, |x: f64| x.to_bits() == 0);

unsafe impl<T: ?Sized> IsZero for *const T {
    #[inline]
    fn is_zero(&self) -> bool {
        (*self).is_null()
    }
}

unsafe impl<T: ?Sized> IsZero for *mut T {
    #[inline]
    fn is_zero(&self) -> bool {
        (*self).is_null()
    }
}


////////////////////////////////////////////////////////////////////////////////
// Common trait implementations for Vec
////////////////////////////////////////////////////////////////////////////////

impl<'a, T: Clone, A: AllocRef<'a>> Clone for Vec<'a, T, A> {
    fn clone(&self) -> Self {
        let cap = self.buf.cap();
        let len = self.len;
        let alloc_ref = self.buf.alloc_ref().clone();
        let buf = RawVec::with_capacity_and_alloc_ref(cap, alloc_ref);
        let ptr = buf.ptr();
        unsafe { ptr::copy_nonoverlapping(self.buf.ptr(), ptr, cap); }
        Vec {
            buf,
            len,
        }
    }
}

impl<'a, T: Hash, A: AllocRef<'a>> Hash for Vec<'a, T, A> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state)
    }
}

impl<'a, T, I: SliceIndex<[T]>, A: AllocRef<'a>> Index<I> for Vec<'a, T, A> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&**self, index)
    }
}

impl<'a, T, I: SliceIndex<[T]>, A: AllocRef<'a>> IndexMut<I> for Vec<'a, T, A> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut **self, index)
    }
}

impl<'a, T, A: AllocRef<'a>> ops::Deref for Vec<'a, T, A> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            let p = self.buf.ptr();
            assume(!p.is_null());
            slice::from_raw_parts(p, self.len)
        }
    }
}

impl<'a, T, A: AllocRef<'a>> ops::DerefMut for Vec<'a, T, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            let ptr = self.buf.ptr();
            assume(!ptr.is_null());
            slice::from_raw_parts_mut(ptr, self.len)
        }
    }
}

impl<'a, T, A: AllocRef<'a>> IntoIterator for Vec<'a, T, A> {
    type Item = T;
    type IntoIter = IntoIter<'a, T, A>;

    #[inline]
    fn into_iter(mut self) -> Self::IntoIter {
        unsafe {
            let begin = self.as_mut_ptr();
            assume(!begin.is_null());
            let end = if mem::size_of::<T>() == 0 {
                arith_offset(begin as *const i8, self.len() as isize) as *const T
            } else {
                begin.add(self.len()) as *const T
            };
            let a = self.buf.alloc_ref().clone();
            let cap = self.buf.cap();
            mem::forget(self);
            IntoIter {
                buf: NonNull::new_unchecked(begin),
                phantom: PhantomData,
                cap,
                ptr: begin,
                end,
                a,
                _a: PhantomData,
            }
        }
    }
}

impl<'a, 'b: 'a, T, A: AllocRef<'a>> IntoIterator for &'b Vec<'a, T, A> {
    type Item = &'b T;
    type IntoIter = slice::Iter<'b, T>;

    fn into_iter(self) -> slice::Iter<'b, T> {
        self.iter()
    }
}

impl<'a, 'b: 'a, T, A: AllocRef<'a>> IntoIterator for &'b mut Vec<'a, T, A> {
    type Item = &'b mut T;
    type IntoIter = slice::IterMut<'b, T>;

    fn into_iter(self) -> slice::IterMut<'b, T> {
        self.iter_mut()
    }
}

impl<'a, T, A: AllocRef<'a>> Extend<T> for Vec<'a, T, A> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        <Self as SpecExtend<T, I::IntoIter>>::spec_extend(self, iter.into_iter())
    }
}

// Specialization trait used for Vec::extend
trait SpecExtend<T, I> {
    fn spec_extend(&mut self, iter: I);
}

impl<'a, T, I, A: AllocRef<'a>> SpecExtend<T, I> for Vec<'a, T, A>
    where I: Iterator<Item=T>,
{
    default fn spec_extend(&mut self, iter: I) {
        self.extend_desugared(iter)
    }
}

impl<'a, T, I, A: AllocRef<'a>> SpecExtend<T, I> for Vec<'a, T, A>
    where I: TrustedLen<Item=T>,
{
    default fn spec_extend(&mut self, iterator: I) {
        // This is the case for a TrustedLen iterator.
        let (low, high) = iterator.size_hint();
        if let Some(high_value) = high {
            debug_assert_eq!(low, high_value,
                             "TrustedLen iterator's size hint is not exact: {:?}",
                             (low, high));
        }
        if let Some(additional) = high {
            self.reserve(additional);
            unsafe {
                let mut ptr = self.as_mut_ptr().add(self.len());
                let mut local_len = SetLenOnDrop::new(&mut self.len);
                iterator.for_each(move |element| {
                    ptr::write(ptr, element);
                    ptr = ptr.offset(1);
                    // NB can't overflow since we would have had to alloc the address space
                    local_len.increment_len(1);
                });
            }
        } else {
            self.extend_desugared(iterator)
        }
    }
}

impl<'a, T, A: AllocRef<'a>> SpecExtend<T, IntoIter<'a, T, A>> for Vec<'a, T, A> {
    fn spec_extend(&mut self, mut iterator: IntoIter<'a, T, A>) {
        unsafe {
            self.append_elements(iterator.as_slice() as _);
        }
        iterator.ptr = iterator.end;
    }
}

impl<'a, 'b: 'a, T: 'b, I, A: AllocRef<'a>> SpecExtend<&'b T, I> for Vec<'a, T, A>
    where I: Iterator<Item=&'b T>,
          T: Clone,
{
    default fn spec_extend(&mut self, iterator: I) {
        self.spec_extend(iterator.cloned())
    }
}

impl<'a, 'b: 'a, T: 'b, A: AllocRef<'a>> SpecExtend<&'b T, slice::Iter<'b, T>> for Vec<'a, T, A>
    where T: Copy,
{
    fn spec_extend(&mut self, iterator: slice::Iter<'b, T>) {
        let slice = iterator.as_slice();
        self.reserve(slice.len());
        unsafe {
            let len = self.len();
            self.set_len(len + slice.len());
            self.get_unchecked_mut(len..).copy_from_slice(slice);
        }
    }
}

impl<'a, T, A: AllocRef<'a>> Vec<'a, T, A> {
    fn extend_desugared<I: Iterator<Item = T>>(&mut self, mut iterator: I) {
        // This is the case for a general iterator.
        //
        // This function should be the moral equivalent of:
        //
        //      for item in iterator {
        //          self.push(item);
        //      }
        while let Some(element) = iterator.next() {
            let len = self.len();
            if len == self.capacity() {
                let (lower, _) = iterator.size_hint();
                self.reserve(lower.saturating_add(1));
            }
            unsafe {
                ptr::write(self.get_unchecked_mut(len), element);
                // NB can't overflow since we would have had to alloc the address space
                self.set_len(len + 1);
            }
        }
    }
}

/// Extend implementation that copies elements out of references before pushing them onto the Vec.
///
/// This implementation is specialized for slice iterators, where it uses [`copy_from_slice`] to
/// append the entire slice at once.
///
/// [`copy_from_slice`]: ../../std/primitive.slice.html#method.copy_from_slice
impl<'a, 'b: 'a, T: 'b + Copy, A: AllocRef<'a>> Extend<&'b T> for Vec<'a, T, A> {
    fn extend<I: IntoIterator<Item = &'b T>>(&mut self, iter: I) {
        self.spec_extend(iter.into_iter())
    }
}

impl <'a, 'b, T: Sized, U, A: AllocRef<'a>, B: AllocRef<'b>> PartialEq<Vec<'b, U, B>> for Vec<'a, T, A> where T: PartialEq<U> {
    #[inline]
    fn eq(&self, other: &Vec<'b, U, B>) -> bool { self[..] == other[..] }

    #[inline]
    fn ne(&self, other: &Vec<'b, U, B>) -> bool { self[..] != other[..] }
}
impl <'a, 'b, T: Sized, U, A: AllocRef<'a>> PartialEq<&'b [U]> for Vec<'a, T, A> where T: PartialEq<U> {
    #[inline]
    fn eq(&self, other: &&'b [U]) -> bool { self[..] == other[..] }

    #[inline]
    fn ne(&self, other: &&'b [U]) -> bool { self[..] != other[..] }
}
impl <'a, 'b, T: Sized, U, A: AllocRef<'a>> PartialEq<&'b mut [U]> for Vec<'a, T, A> where T: PartialEq<U> {
    #[inline]
    fn eq(&self, other: &&'b mut [U]) -> bool { self[..] == other[..] }

    #[inline]
    fn ne(&self, other: &&'b mut [U]) -> bool { self[..] != other[..] }
}


macro_rules! array_impls {
    ($($N: expr)+) => {
        $(
            impl<'a, T: Sized, U, A: AllocRef<'a>> PartialEq<[U; $N]> for Vec<'a, T, A> where T: PartialEq<U> {
                #[inline]
                fn eq(&self, other: &[U; $N]) -> bool { self[..] == other[..] }

                #[inline]
                fn ne(&self, other: &[U; $N]) -> bool { self[..] != other[..] }
            }
            impl<'a, 'b, T: Sized, U, A: AllocRef<'a>> PartialEq<&'b [U; $N]> for Vec<'a, T, A> where T: PartialEq<U> {
                #[inline]
                fn eq(&self, other: &&'b [U; $N]) -> bool { self[..] == other[..] }

                #[inline]
                fn ne(&self, other: &&'b [U; $N]) -> bool { self[..] != other[..] }
            }
            // NOTE: some less important impls are omitted to reduce code bloat
            // __impl_slice_eq1! { Vec<A>, &'b mut [B; $N] }
            // __impl_slice_eq1! { Cow<'a, [A]>, [B; $N], Clone }
            // __impl_slice_eq1! { Cow<'a, [A]>, &'b [B; $N], Clone }
            // __impl_slice_eq1! { Cow<'a, [A]>, &'b mut [B; $N], Clone }
        )+
    }
}

array_impls! {
     0  1  2  3  4  5  6  7  8  9
    10 11 12 13 14 15 16 17 18 19
    20 21 22 23 24 25 26 27 28 29
    30 31 32
}

/// Implements comparison of vectors, lexicographically.
impl<'a, T: PartialOrd, A: AllocRef<'a>> PartialOrd for Vec<'a, T, A> {
    #[inline]
    fn partial_cmp(&self, other: &Vec<'a, T, A>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<'a, T: Eq, A: AllocRef<'a>> Eq for Vec<'a, T, A> {}

/// Implements ordering of vectors, lexicographically.
impl<'a, T: Ord, A: AllocRef<'a>> Ord for Vec<'a, T, A> {
    #[inline]
    fn cmp(&self, other: &Vec<'a, T, A>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

unsafe impl<'a, #[may_dangle] T, A: AllocRef<'a>> Drop for Vec<'a, T, A> {
    fn drop(&mut self) {
        unsafe {
            // use drop for [T]
            ptr::drop_in_place(&mut self[..]);
        }
        // RawVec handles deallocation
    }
}

impl<'a, T: fmt::Debug, A: AllocRef<'a>> fmt::Debug for Vec<'a, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<'a, T, A: AllocRef<'a>> AsRef<Vec<'a, T, A>> for Vec<'a, T, A> {
    fn as_ref(&self) -> &Vec<'a, T, A> {
        self
    }
}

impl<'a, T, A: AllocRef<'a>> AsMut<Vec<'a, T, A>> for Vec<'a, T, A> {
    fn as_mut(&mut self) -> &mut Vec<'a, T, A> {
        self
    }
}

impl<'a, T, A: AllocRef<'a>> AsRef<[T]> for Vec<'a, T, A> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<'a, T, A: AllocRef<'a>> AsMut<[T]> for Vec<'a, T, A> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<'a, A: AllocRef<'a>> From<Box<'a, str, A>> for Vec<'a, u8, A> {
    fn from(s: Box<'a, str, A>) -> Vec<'a, u8, A> {
        let bytes = s.as_bytes();
        let cap = bytes.len();
        let (ptr, a) = Box::into_raw(s);
        unsafe { Vec::from_raw_parts(ptr as *mut u8, cap, cap, a) }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Iterators
////////////////////////////////////////////////////////////////////////////////

/// An iterator that moves out of a vector.
///
/// This `struct` is created by the `into_iter` method on [`Vec`][`Vec`] (provided
/// by the [`IntoIterator`] trait).
///
/// [`Vec`]: struct.Vec.html
/// [`IntoIterator`]: ../../std/iter/trait.IntoIterator.html
pub struct IntoIter<'a, T, A: AllocRef<'a>> {
    buf: NonNull<T>,
    phantom: PhantomData<T>,
    cap: usize,
    ptr: *const T,
    end: *const T,
    a: A,
    _a: PhantomData<&'a A>,
}

impl<'a, T: fmt::Debug, A: AllocRef<'a>> fmt::Debug for IntoIter<'a, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter")
            .field(&self.as_slice())
            .finish()
    }
}

impl<'a, T, A: AllocRef<'a>> IntoIter<'a, T, A> {
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.ptr, self.len())
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(self.ptr as *mut T, self.len())
        }
    }
}

unsafe impl<'a, T: Send, A: AllocRef<'a>> Send for IntoIter<'a, T, A> {}
unsafe impl<'a, T: Sync, A: AllocRef<'a>> Sync for IntoIter<'a, T, A> {}

impl<'a, T, A: AllocRef<'a>> Iterator for IntoIter<'a, T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe {
            if self.ptr as *const _ == self.end {
                None
            } else {
                if mem::size_of::<T>() == 0 {
                    // purposefully don't use 'ptr.offset' because for
                    // vectors with 0-size elements this would return the
                    // same pointer.
                    self.ptr = arith_offset(self.ptr as *const i8, 1) as *mut T;

                    // Make up a value of this ZST.
                    Some(mem::zeroed())
                } else {
                    let old = self.ptr;
                    self.ptr = self.ptr.offset(1);

                    Some(ptr::read(old))
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = if mem::size_of::<T>() == 0 {
            (self.end as usize).wrapping_sub(self.ptr as usize)
        } else {
            unsafe { self.end.offset_from(self.ptr) as usize }
        };
        (exact, Some(exact))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

impl<'a, T, A: AllocRef<'a>> DoubleEndedIterator for IntoIter<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        unsafe {
            if self.end == self.ptr {
                None
            } else {
                if mem::size_of::<T>() == 0 {
                    // See above for why 'ptr.offset' isn't used
                    self.end = arith_offset(self.end as *const i8, -1) as *mut T;

                    // Make up a value of this ZST.
                    Some(mem::zeroed())
                } else {
                    self.end = self.end.offset(-1);

                    Some(ptr::read(self.end))
                }
            }
        }
    }
}

impl<'a, T, A: AllocRef<'a>> ExactSizeIterator for IntoIter<'a, T, A> {
    fn is_empty(&self) -> bool {
        self.ptr == self.end
    }
}

impl<'a, T, A: AllocRef<'a>> FusedIterator for IntoIter<'a, T, A> {}

unsafe impl<'a, T, A: AllocRef<'a>> TrustedLen for IntoIter<'a, T, A> {}

unsafe impl<'a, #[may_dangle] T, A: AllocRef<'a>> Drop for IntoIter<'a, T, A> {
    fn drop(&mut self) {
        // destroy the remaining elements
        for _x in self.by_ref() {}

        // RawVec handles deallocation
        let alloc_ref = self.a.clone();
        let _ = unsafe { RawVec::from_raw_parts_and_alloc_ref(self.buf.as_ptr(), self.cap, alloc_ref) };
    }
}