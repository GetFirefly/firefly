#![allow(unused)]
use core::borrow;
use core::cmp::Ordering;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
///! A reimplementation of `Box<T>` from the standard library that takes
///! an allocator reference allowing one to control which heap the Box allocates
///! on.
///!
///! See `Box<T>` for usage and documentation
use core::mem;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::ptr::{self, NonNull, Unique};
use core::str;

use core_alloc::slice;

use crate::alloc::{AllocRef, Layout};
use crate::alloc::alloc_handle::{AllocHandle, AsAllocHandle};

/// This an alternative implementation of `Box` that
/// supports the use of a custom allocator in place
/// of the global allocator
pub struct Box<'a, T: ?Sized, A: AllocHandle<'a>> {
    t: Unique<T>,
    a: A,
    _phantom: PhantomData<&'a A>,
}

impl<'a, T, A: ?Sized + AllocRef + Sync, H: AllocHandle<'a, AllocRef = A>> Box<'a, T, H> {
    /// Creates a new `Box` by moving the given value on to the heap,
    /// the memory is allocated via the provided allocator
    #[allow(unused)]
    #[inline]
    pub fn new<R: 'a + AsAllocHandle<'a, Handle = H>>(t: T, a: &'a R) -> Self {
        // Move the value into the allocator's heap
        let mut alloc_handle = a.as_alloc_handle();
        let layout = Layout::for_value(&t);
        let ptr = unsafe {
            alloc_handle
                .alloc(layout)
                .map(|(ptr, _)| ptr)
                .expect("failed to allocate Box")
                .cast()
        };
        let raw = ptr.as_ptr();
        unsafe { ptr::write(raw as *mut _, t) };
        Self {
            t: unsafe { Unique::new_unchecked(raw) },
            a: alloc_handle,
            _phantom: PhantomData,
        }
    }
}

impl<'a, A: ?Sized + AllocRef + Sync, H: AllocHandle<'a, AllocRef = A>> Box<'a, str, H> {
    /// Constructs a `Box<str>` from a `&str`, which
    /// can then be used in place of `&str` for heap allocated
    /// strings
    #[inline]
    pub fn from_str<R: 'a + AsAllocHandle<'a, Handle = H>>(s: &str, a: &'a R) -> Self {
        let mut alloc_handle = a.as_alloc_handle();
        let len = s.len();
        let ptr = unsafe {
            let layout = Layout::from_size_align_unchecked(len, mem::size_of::<usize>());
            alloc_handle
                .alloc(layout)
                .map(|(ptr, _)| ptr)
                .expect("failed to allocate Box from str")
                .as_ptr()
        };
        unsafe { ptr::copy_nonoverlapping(s.as_ptr(), ptr, len) };
        let bytes = unsafe { slice::from_raw_parts(ptr, len) };
        let string = unsafe { str::from_utf8_unchecked(bytes) as *const str as *mut str };
        Self {
            t: unsafe { Unique::new_unchecked(string) },
            a: alloc_handle,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: ?Sized, A: AllocHandle<'a>> Box<'a, T, A> {
    #[inline]
    pub unsafe fn from_raw(raw: *mut T, a: A) -> Self {
        Self {
            t: Unique::new_unchecked(raw),
            a,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn into_raw(b: Box<'a, T, A>) -> (*mut T, A) {
        let (non_null, alloc_handle) = Box::into_raw_non_null(b);
        (non_null.as_ptr(), alloc_handle)
    }

    #[inline]
    pub fn into_raw_non_null(b: Box<'a, T, A>) -> (NonNull<T>, A) {
        let (uniq, alloc_handle) = Box::into_unique(b);
        (uniq.into(), alloc_handle)
    }

    #[inline]
    #[doc(hidden)]
    pub fn into_unique(mut b: Box<'a, T, A>) -> (Unique<T>, A) {
        // Box is kind-of a library type, but recognized as a "unique pointer" by
        // Stacked Borrows.  This function here corresponds to "reborrowing to
        // a raw pointer", but there is no actual reborrow here -- so
        // without some care, the pointer we are returning here still carries
        // the `Uniq` tag.  We round-trip through a mutable reference to avoid that.
        let alloc_handle = b.a.clone();
        let unique = unsafe { b.t.as_mut() as *mut T };
        mem::forget(b);
        (unsafe { Unique::new_unchecked(unique) }, alloc_handle)
    }

    #[inline]
    pub fn leak(b: Box<'a, T, A>) -> (&'a mut T, A)
    where
        T: 'a, // Technically not needed, but kept to be explicit.
    {
        unsafe {
            let (ptr, alloc_handle) = Box::into_raw(b);
            (&mut *ptr, alloc_handle)
        }
    }

    /// Converts a `Box<T, A>` into a `Pin<Box<T, A>>`
    ///
    /// This conversion does not allocate on the heap and happens in place.
    ///
    /// This is also available via [`From`].
    pub fn into_pin(boxed: Box<'a, T, A>) -> Pin<Box<'a, T, A>> {
        // It's not possible to move or replace the insides of a `Pin<Box<T, A>>`
        // when `T: !Unpin`,  so it's safe to pin it directly without any
        // additional requirements.
        unsafe { Pin::new_unchecked(boxed) }
    }
}

impl<'a, T: ?Sized, A: AllocHandle<'a>> Deref for Box<'a, T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { self.t.as_ref() }
    }
}
impl<'a, T: ?Sized, A: AllocHandle<'a>> DerefMut for Box<'a, T, A> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { self.t.as_mut() }
    }
}
impl<'a, T: ?Sized, A: AllocHandle<'a>> borrow::Borrow<T> for Box<'a, T, A> {
    fn borrow(&self) -> &T {
        unsafe { self.t.as_ref() }
    }
}
impl<'a, T: ?Sized, A: AllocHandle<'a>> borrow::BorrowMut<T> for Box<'a, T, A> {
    fn borrow_mut(&mut self) -> &mut T {
        unsafe { self.t.as_mut() }
    }
}
impl<'a, T: ?Sized, A: AllocHandle<'a>> Drop for Box<'a, T, A> {
    fn drop(&mut self) {
        let t = unsafe { self.t.as_ref() };
        let layout = Layout::for_value(t);
        let non_null = unsafe { NonNull::new_unchecked(self.t.as_mut() as *mut _ as *mut u8) };
        unsafe { self.a.dealloc(non_null, layout) };
    }
}
impl<'a, T: ?Sized + PartialEq, A: AllocHandle<'a>> PartialEq for Box<'a, T, A> {
    #[inline]
    fn eq(&self, other: &Box<'a, T, A>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
    #[inline]
    fn ne(&self, other: &Box<'a, T, A>) -> bool {
        PartialEq::ne(&**self, &**other)
    }
}
impl<'a, T: ?Sized + PartialOrd, A: AllocHandle<'a>> PartialOrd for Box<'a, T, A> {
    #[inline]
    fn partial_cmp(&self, other: &Box<'a, T, A>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
    #[inline]
    fn lt(&self, other: &Box<'a, T, A>) -> bool {
        PartialOrd::lt(&**self, &**other)
    }
    #[inline]
    fn le(&self, other: &Box<'a, T, A>) -> bool {
        PartialOrd::le(&**self, &**other)
    }
    #[inline]
    fn ge(&self, other: &Box<'a, T, A>) -> bool {
        PartialOrd::ge(&**self, &**other)
    }
    #[inline]
    fn gt(&self, other: &Box<'a, T, A>) -> bool {
        PartialOrd::gt(&**self, &**other)
    }
}
impl<'a, T: ?Sized + Ord, A: AllocHandle<'a>> Ord for Box<'a, T, A> {
    #[inline]
    fn cmp(&self, other: &Box<'a, T, A>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}
impl<'a, T: ?Sized + Eq, A: AllocHandle<'a>> Eq for Box<'a, T, A> {}

impl<'a, T: ?Sized + Hash, A: AllocHandle<'a>> Hash for Box<'a, T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<'a, T: ?Sized + Hasher, A: AllocHandle<'a>> Hasher for Box<'a, T, A> {
    fn finish(&self) -> u64 {
        (**self).finish()
    }
    fn write(&mut self, bytes: &[u8]) {
        (**self).write(bytes)
    }
    fn write_u8(&mut self, i: u8) {
        (**self).write_u8(i)
    }
    fn write_u16(&mut self, i: u16) {
        (**self).write_u16(i)
    }
    fn write_u32(&mut self, i: u32) {
        (**self).write_u32(i)
    }
    fn write_u64(&mut self, i: u64) {
        (**self).write_u64(i)
    }
    fn write_u128(&mut self, i: u128) {
        (**self).write_u128(i)
    }
    fn write_usize(&mut self, i: usize) {
        (**self).write_usize(i)
    }
    fn write_i8(&mut self, i: i8) {
        (**self).write_i8(i)
    }
    fn write_i16(&mut self, i: i16) {
        (**self).write_i16(i)
    }
    fn write_i32(&mut self, i: i32) {
        (**self).write_i32(i)
    }
    fn write_i64(&mut self, i: i64) {
        (**self).write_i64(i)
    }
    fn write_i128(&mut self, i: i128) {
        (**self).write_i128(i)
    }
    fn write_isize(&mut self, i: isize) {
        (**self).write_isize(i)
    }
}
