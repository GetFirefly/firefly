use core::arch::asm;

use liblumen_rt::term::AtomData;

extern "C-unwind" {
    /// This function is defined in `liblumen_alloc::erts::term::atom`
    #[link_name = "__lumen_initialize_atom_table"]
    pub fn init(start: *const AtomData, end: *const AtomData) -> bool;
}


#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn start() -> *const AtomData {
    let mut ptr: *mut AtomData = core::ptr::null_mut();
    unsafe {
        asm!(
            "adrp {x}, section$start$__DATA$__atoms@GOTPAGE",
            "ldr {x}, [{x}, section$start$__DATA$__atoms@GOTPAGEOFF]",
            x = inout(reg) ptr,
            options(readonly, preserves_flags, nostack)
        );
    }
    ptr
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(super) fn start() -> *const AtomData {
    let mut ptr: *mut AtomData = core::ptr::null_mut();
    unsafe {
        asm!(
            "mov {x}, [rip + section$start$__DATA$__atoms@GOTPCREL]",
            x = inout(reg) ptr,
            options(readonly, preserves_flags, nostack)
        );
    }
    ptr
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn end() -> *const AtomData {
    let mut ptr: *mut AtomData = core::ptr::null_mut();
    unsafe {
        asm!(
            "adrp {x}, section$end$__DATA$__atoms@GOTPAGE",
            "ldr {x}, [{x}, section$end$__DATA$__atoms@GOTPAGEOFF]",
            x = inout(reg) ptr,
            options(readonly, preserves_flags, nostack)
        );
    }
    ptr
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(super) fn end() -> *const AtomData {
    let mut ptr: *mut AtomData = core::ptr::null_mut();
    unsafe {
        asm!(
            "mov {x}, [rip + section$end$__DATA$__atoms@GOTPCREL]",
            x = inout(reg) ptr,
            options(readonly, preserves_flags, nostack)
        );
    }
    ptr
}
