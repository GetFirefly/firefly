use core::ptr::{self, NonNull};
use core::alloc::{GlobalAlloc, Layout, AllocErr};

use crate::sys_alloc::SysAlloc;

static mut DLMALLOC: dlmalloc::Dlmalloc = dlmalloc::DLMALLOC_INIT;

#[inline]
pub unsafe fn alloc(layout: Layout) -> Result<NonNull<u8>, AllocErr> {
    let _lock = lock::lock();
    let ptr = DLMALLOC.malloc(layout.size(), layout.align());
    NonNull::new(ptr).ok_or(AllocErr)
}

#[inline]
pub unsafe fn alloc_zeroed(layout: Layout) -> Result<NonNull<u8>, AllocErr> {
    let _lock = lock::lock();
    let ptr = DLMALLOC.calloc(layout.size(), layout.align());
    NonNull::new(ptr).ok_or(AllocErr)
}

#[inline]
pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> Result<NonNull<u8>, AllocErr> {
    let _lock = lock::lock();
    let new_ptr = DLMALLOC.realloc(ptr, layout.size(), layout.align(), new_size);
    NonNull::new(new_ptr).ok_or(AllocErr)
}

#[inline]
pub unsafe fn free(ptr: *mut u8, layout: Layout) {
    let _lock = lock::lock();
    DLMALLOC.free(ptr, layout.size(), layout.align());
}

unsafe impl GlobalAlloc for SysAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self::alloc(layout).unwrap_or(ptr::null_mut())
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self::alloc_zeroed(layout).unwrap_or(ptr::null_mut())
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self::free(ptr, layout)
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self::realloc(ptr, layout, new_size).unwrap_or(ptr::null_mut())
    }
}

#[cfg(target_feature = "atomics")]
mod lock {
    use core::arch::wasm32;
    use core::sync::atomic::{AtomicI32, Ordering::SeqCst};

    static LOCKED: AtomicI32 = AtomicI32::new(0);

    pub struct DropLock;

    pub fn lock() -> DropLock {
        loop {
            if LOCKED.swap(1, SeqCst) == 0 {
                return DropLock
            }
            unsafe {
                let r = wasm32::i32_atomic_wait(
                    &LOCKED as *const AtomicI32 as *mut i32,
                    1,  // expected value
                    -1, // timeout
                );
                debug_assert!(r == 0 || r == 1);
            }
        }
    }

    impl Drop for DropLock {
        fn drop(&mut self) {
            let r = LOCKED.swap(0, SeqCst);
            debug_assert_eq!(r, 1);
            unsafe {
                wasm32::atomic_notify(
                    &LOCKED as *const AtomicI32 as *mut i32,
                    1, // only one thread
                );
            }
        }
    }
}

#[cfg(not(target_feature = "atomics"))]
mod lock {
    #[inline]
    pub fn lock() {}
}
