use core::convert::{AsMut, AsRef};
use core::ops::{Deref, DerefMut};

/// This type wraps another type, guaranteeing that it is
/// aligned to a cache line boundary. This is useful for
/// preventing false sharing/ping ponging between core caches.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(align(64))]
pub struct CachePadded<T> {
    inner: T,
}
impl<T> CachePadded<T> {
    /// Create a new cache-padded value
    pub const fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Unwrap the inner value, consuming the wrapper in the process
    pub fn unwrap(self) -> T {
        self.inner
    }
}
impl<T: Copy> Copy for CachePadded<T> {}
impl<T> AsRef<T> for CachePadded<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.inner
    }
}
impl<T> AsMut<T> for CachePadded<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}
impl<T> Deref for CachePadded<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<T> DerefMut for CachePadded<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
