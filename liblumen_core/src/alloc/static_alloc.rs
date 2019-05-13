use core::alloc::Alloc;

pub unsafe trait StaticAlloc: Alloc + Sync {
    unsafe fn static_ref() -> &'static Self;
    unsafe fn static_mut() -> &'static mut Self;
}