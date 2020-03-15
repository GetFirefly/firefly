use core::cell::UnsafeCell;
use core::fmt;
use core::mem;
use core::ops::Deref;

/// A `Cell`-like type for use in thread local variables
///
/// In a thread-local context, we do not need to worry about
/// concurrent borrows/mutation, since nothing else can possibly
/// reference the same value at the same time.
///
/// By default, the standard library only provides `RefCell` as a
/// type that can live in a thread local while permitting mutation,
/// but it requires some overhead as it does runtime borrow checking.
/// For us, by far the most common case is that we access a given thread
/// local within a single function, oftentimes in a hot path, so incurring
/// that overhead is not desirable. This type allows us to work with thread
/// locals in a more natural, and perf-friendly way.
///
/// ## Safety
///
/// This type is _not_ safe to use in any situation other than a thread local
/// context. It deliberately omits safety checks by relying on guarantees provided
/// by the nature of a thread local environment; but if used elsewhere, it comes
/// with an enormous risk of incurring undefined behavior.
pub struct ThreadLocalCell<T>(UnsafeCell<T>);

impl<T> ThreadLocalCell<T> {
    pub const fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    #[inline]
    pub fn as_ref(&self) -> &T {
        unsafe { &*(self.0.get() as *const _) }
    }

    #[inline]
    pub unsafe fn as_mut<'this, 'a: 'this>(&'this self) -> &'a mut T {
        &mut *self.0.get()
    }

    #[inline]
    pub unsafe fn set(&self, value: T) {
        *self.0.get() = value;
    }

    #[inline]
    pub unsafe fn replace(&self, value: T) -> T {
        mem::replace(&mut *self.0.get(), value)
    }
}

impl<T> Deref for ThreadLocalCell<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T: Default> Default for ThreadLocalCell<T> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}
impl<T: Clone> Clone for ThreadLocalCell<T> {
    fn clone(&self) -> Self {
        Self::new(self.as_ref().clone())
    }
}
impl<T: fmt::Debug> fmt::Debug for ThreadLocalCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.as_ref())
    }
}
impl<T: fmt::Display> fmt::Display for ThreadLocalCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}
