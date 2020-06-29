use liblumen_core::symbols::FunctionSymbol;

extern "C" {
    /// This symbol is defined in the compiled executable,
    /// and specifies the number of symbols in the symbol table.
    #[link_name = "__LUMEN_SYMBOL_TABLE_SIZE"]
    pub static NUM_SYMBOLS: usize;

    /// This symbol is defined in the compiled executable,
    /// and provides a pointer to the symbol table, or more specifically,
    /// a pointer to the first pointer in the symbol table. The symbol table
    /// is an array of pointers to FunctionSymbol structs, each of which
    /// contains the symbol itself as well as an opaque function pointer.
    ///
    /// Combined with `NUM_SYMBOLS`, this can be used to obtain a slice
    /// of pointers.
    #[link_name = "__LUMEN_SYMBOL_TABLE"]
    pub static SYMBOL_TABLE: *const FunctionSymbol;

    /// This function is defined in `liblumen_alloc::erts::apply`
    pub fn InitializeLumenDispatchTable(table: *const FunctionSymbol, len: usize) -> bool;
}
