use core::cmp;
use core::convert::Into;
use core::fmt;
use core::hash;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::erts::exception::InternalResult;

use super::encoding::{self, Boxable, UnsizedBoxable};
use super::prelude::Term;

/// Represents boxed terms.
///
/// A `Boxed<T>` is designed around being reified from a raw
/// pointer to a typed term with `Encoded::decode`, and otherwise
/// only being consumed
#[repr(transparent)]
pub struct Boxed<T: ?Sized>(NonNull<T>);
impl<T: Sized> Boxed<T> {
    /// Creates a new `Boxed` that is dangling, but well-aligned.
    ///
    /// This is useful for initializing types which lazily allocate, like
    /// `Vec::new` does.
    ///
    /// Note that the pointer value may potentially represent a valid pointer to
    /// a `T`, which means this must not be used as a "not yet initialized"
    /// sentinel value. Types that lazily allocate must track initialization by
    /// some other means.
    #[inline]
    pub const fn dangling() -> Self {
        Self(NonNull::dangling())
    }
}
impl<T: ?Sized> Boxed<T> {
    /// Creates a new `Boxed` if `ptr` is non-null.
    #[inline]
    pub fn new(ptr: *mut T) -> Option<Self> {
        if !ptr.is_null() {
            Some(Self(unsafe { NonNull::new_unchecked(ptr) }))
        } else {
            None
        }
    }

    /// Creates a new `Boxed`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null.
    #[inline(always)]
    pub const unsafe fn new_unchecked(ptr: *mut T) -> Self {
        Self(NonNull::new_unchecked(ptr))
    }

    /// Acquires the underlying `*mut` pointer.
    #[inline(always)]
    pub const fn as_ptr(self) -> *mut T {
        self.0.as_ptr()
    }

    /// Dereferences the content.
    ///
    /// The resulting lifetime is bound to self so this behaves "as if"
    /// it were actually an instance of T that is getting borrowed. If a longer
    /// (unbound) lifetime is needed, use `&*my_ptr.as_ptr()`.
    #[inline(always)]
    pub fn as_ref(&self) -> &T {
        unsafe { self.0.as_ref() }
    }

    /// Mutably dereferences the content.
    ///
    /// The resulting lifetime is bound to self so this behaves "as if"
    /// it were actually an instance of T that is getting borrowed. If a longer
    /// (unbound) lifetime is needed, use `&mut *my_ptr.as_ptr()`.
    #[inline(always)]
    pub fn as_mut(&mut self) -> &mut T {
        unsafe { self.0.as_mut() }
    }

    /// Casts to a pointer of another type.
    #[inline(always)]
    pub const fn cast<U>(self) -> Boxed<U> {
        Boxed(self.0.cast::<U>())
    }
}
impl<T: ?Sized> Clone for Boxed<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// `NonNull` pointers are not `Send` because the data they reference may be aliased.
// N.B., this impl is unnecessary, but should provide better error messages.
//impl<T: ?Sized> !Send for Boxed<T> { }

/// `NonNull` pointers are not `Sync` because the data they reference may be aliased.
// N.B., this impl is unnecessary, but should provide better error messages.
//impl<T: ?Sized> !Sync for Boxed<T> { }

impl<T: ?Sized> Copy for Boxed<T> {}

impl<T: ?Sized> fmt::Debug for Boxed<T> {
    default fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ty = core::any::type_name::<T>();
        write!(f, "Boxed<{}>({:p})", ty, self.as_ptr())
    }
}
impl<T: ?Sized> fmt::Debug for Boxed<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ty = core::any::type_name::<T>();
        let inner = self.as_ref();
        write!(f, "Boxed<{}>({:?} at {:p})", ty, inner, self.as_ptr())
    }
}

impl<T: ?Sized> fmt::Pointer for Boxed<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.as_ptr(), f)
    }
}

impl<T> Eq for Boxed<T> where T: ?Sized + PartialEq<Boxed<T>> {}

impl<T, U> PartialEq<U> for Boxed<T>
where
    T: ?Sized + PartialEq<U>,
    U: ?Sized,
{
    #[inline]
    default fn eq(&self, other: &U) -> bool {
        self.as_ref().eq(other)
    }
}

impl<T> Ord for Boxed<T>
where
    T: ?Sized + PartialOrd<Boxed<T>> + Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.as_ref().cmp(other.as_ref())
    }
}

impl<T, U> PartialOrd<U> for Boxed<T>
where
    T: ?Sized + PartialOrd<U>,
    U: ?Sized,
{
    #[inline]
    fn partial_cmp(&self, other: &U) -> Option<cmp::Ordering> {
        self.as_ref().partial_cmp(other)
    }
}

impl<T> hash::Hash for Boxed<T>
where
    T: ?Sized + hash::Hash,
{
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

impl<T: ?Sized> From<&mut T> for Boxed<T> {
    #[inline]
    fn from(reference: &mut T) -> Self {
        Self(unsafe { NonNull::new_unchecked(reference as *const _ as *mut T) })
    }
}

impl<T: ?Sized> From<&T> for Boxed<T> {
    #[inline]
    fn from(reference: &T) -> Self {
        Self(unsafe { NonNull::new_unchecked(reference as *const _ as *mut T) })
    }
}

impl<T: ?Sized> From<NonNull<T>> for Boxed<T> {
    #[inline]
    fn from(reference: NonNull<T>) -> Self {
        Self(reference)
    }
}

impl<T: ?Sized> Into<NonNull<T>> for Boxed<T> {
    #[inline]
    fn into(self) -> NonNull<T> {
        self.0
    }
}

impl<T> From<*mut Term> for Boxed<T>
where
    T: UnsizedBoxable<Term>,
{
    default fn from(ptr: *mut Term) -> Boxed<T> {
        assert_ne!(ptr, core::ptr::null_mut());
        unsafe { T::from_raw_term(ptr) }
    }
}

impl<T> From<*mut Term> for Boxed<T>
where
    T: Boxable<Term>,
{
    #[inline]
    default fn from(ptr: *mut Term) -> Self {
        assert_ne!(ptr, core::ptr::null_mut());
        unsafe { Boxed::new_unchecked(ptr as *mut T) }
    }
}

impl<T> Into<*mut Term> for Boxed<T>
where
    T: Boxable<Term>,
{
    #[inline]
    default fn into(self) -> *mut Term {
        self.0.cast::<Term>().as_ptr()
    }
}

impl<T> Into<*mut Term> for Boxed<T>
where
    T: UnsizedBoxable<Term>,
{
    #[inline]
    default fn into(self) -> *mut Term {
        self.0.cast::<Term>().as_ptr()
    }
}

impl<T> Into<*mut T> for Boxed<T> {
    #[inline]
    default fn into(self) -> *mut T {
        self.0.as_ptr()
    }
}

impl<T: ?Sized> Deref for Boxed<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.as_ref()
    }
}

impl<T: ?Sized> DerefMut for Boxed<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.as_mut()
    }
}

impl<T, E> encoding::Encode<E> for Boxed<T>
where
    T: Boxable<Term>,
    E: encoding::Encoded + From<*mut T>,
{
    #[inline]
    fn encode(&self) -> InternalResult<E> {
        Ok(self.as_ptr().into())
    }
}
