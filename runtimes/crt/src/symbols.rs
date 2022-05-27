use core::arch::asm;

use liblumen_rt::function::FunctionSymbol;

extern "C-unwind" {
    #[link_name = "__lumen_initialize_dispatch_table"]
    pub fn init(start: *const FunctionSymbol, end: *const FunctionSymbol) -> bool;
}

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
