use crate::alloc::AllocRef;

pub unsafe trait StaticAlloc: AllocRef + Sync {
    unsafe fn static_ref() -> &'static Self;
    unsafe fn static_mut() -> &'static mut Self;
}
