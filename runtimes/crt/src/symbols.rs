use liblumen_core::symbols::FunctionSymbol;

extern "C" {
    #[link_name = "__lumen_dispatch_start"]
    pub static DISPATCH_START: *const FunctionSymbol;

    #[link_name = "__lumen_dispatch_end"]
    pub static DISPATCH_END: *const FunctionSymbol;

    /// This function is defined in `liblumen_alloc::erts::apply`
    pub fn InitializeLumenDispatchTable(
        start: *const FunctionSymbol,
        end: *const FunctionSymbol,
    ) -> bool;
}
