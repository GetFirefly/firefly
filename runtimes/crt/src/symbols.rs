use core::arch::asm;

use liblumen_rt::function::FunctionSymbol;

extern "C-unwind" {
    #[link_name = "__lumen_initialize_dispatch_table"]
    pub fn init(start: *const FunctionSymbol, end: *const FunctionSymbol) -> bool;
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn start() -> *const FunctionSymbol {
    let mut ptr: *mut FunctionSymbol = core::ptr::null_mut();
    unsafe {
        asm!(
            "adrp {x}, section$start$__DATA$__dispatch@GOTPAGE",
            "ldr {x}, [{x}, section$start$__DATA$__dispatch@GOTPAGEOFF]",
            x = inout(reg) ptr,
            options(readonly, preserves_flags, nostack)
        );
    }
    ptr
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(super) fn start() -> *const FunctionSymbol {
    let mut ptr: *mut FunctionSymbol = core::ptr::null_mut();
    unsafe {
        asm!(
        "mov {x}, [rip + section$start$__DATA$__dispatch@GOTPCREL]",
        x = inout(reg) ptr,
        options(readonly, preserves_flags, nostack)
        );
    }
    ptr
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn end() -> *const FunctionSymbol {
    let mut ptr: *mut FunctionSymbol = core::ptr::null_mut();
    unsafe {
        asm!(
            "adrp {x}, section$end$__DATA$__dispatch@GOTPAGE",
            "ldr {x}, [{x}, section$end$__DATA$__dispatch@GOTPAGEOFF]",
            x = inout(reg) ptr,
            options(readonly, preserves_flags, nostack)
        );
    }
    ptr
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(super) fn end() -> *const FunctionSymbol {
    let mut ptr: *mut FunctionSymbol = core::ptr::null_mut();
    unsafe {
        asm!(
            "mov {x}, [rip + section$end$__DATA$__dispatch@GOTPCREL]",
            x = inout(reg) ptr,
            options(readonly, preserves_flags, nostack)
        );
    }
    ptr
}
