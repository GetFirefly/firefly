use core::convert::{AsMut, AsRef};
use core::ops::{Deref, DerefMut};

/// This type wraps another type, guaranteeing that it is aligned/padded to a cache line boundary.
///
/// This is useful for preventing false sharing/ping ponging between core caches.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
// s390x has 256 byte cache line size
#[cfg_attr(target_arch = "s390x", repr(align(256)))]
// These targets all have 128-byte cache line size
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), repr(align(128)))]
// These targets all have 32-byte cache line size
#[cfg_attr(
    any(
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "riscv64"
    ),
    repr(align(32))
)]
// x86 and Wasm have 64-byte cache-line size, and we assume 64 bytes for all other platforms
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "riscv64"
    )),
    repr(align(64))
)]
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
unsafe impl<T: Sync> Sync for CachePadded<T> {}
unsafe impl<T: Send> Send for CachePadded<T> {}
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
impl<T> From<T> for CachePadded<T> {
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}
