//! This module defines a wrapper type for objects
//! that should be aligned to a cache line to prevent
//! false sharing/ping ponging between core caches
use core::ops::{Deref, DerefMut};

#[repr(align(64))]
pub struct CachePadded<T> {
    inner: T,
}
impl<T> CachePadded<T> {
    /// Create a new `CachePadded<T>`
    pub const fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Unwrap the inner `T`, consuming the wrapper in the process
    pub fn unwrap(self) -> T {
        self.inner
    }
}
impl<T> AsRef<T> for CachePadded<T> {
    fn as_ref(&self) -> &T {
        &self.inner
    }
}
impl<T> Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<T> DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
