use std::sync::Once;

use liblumen_core::symbols::FunctionSymbol;

use liblumen_alloc::erts::apply::InitializeLumenDispatchTable;

pub fn once(function_symbols: &[FunctionSymbol]) {
    let range = function_symbols.as_ptr_range();
    ONCE.call_once(|| {
        unsafe { InitializeLumenDispatchTable(range.start, range.end) };
    });
}

static ONCE: Once = Once::new();
