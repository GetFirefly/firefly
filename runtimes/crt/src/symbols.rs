use firefly_rt::function::FunctionSymbol;

extern "C-unwind" {
    #[link_name = "__firefly_initialize_dispatch_table"]
    pub fn init(start: *const FunctionSymbol, end: *const FunctionSymbol) -> bool;
}

#[cfg(target_os = "macos")]
extern "C" {
    #[link_name = "\x01section$start$__DATA$__dispatch"]
    static DISPATCH_START: FunctionSymbol;

    #[link_name = "\x01section$end$__DATA$__dispatch"]
    static DISPATCH_END: FunctionSymbol;
}

#[cfg(all(unix, not(target_os = "macos")))]
extern "C" {
    #[link_name = "__start___dispatch"]
    static DISPATCH_START: FunctionSymbol;

    #[link_name = "__stop___dispatch"]
    static DISPATCH_END: FunctionSymbol;
}

pub(super) fn start() -> *const FunctionSymbol {
    unsafe { &DISPATCH_START }
}

pub(super) fn end() -> *const FunctionSymbol {
    unsafe { &DISPATCH_END }
}
