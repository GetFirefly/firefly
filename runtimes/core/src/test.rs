use std::sync::Once;

use liblumen_core::symbols::FunctionSymbol;

use liblumen_alloc::erts::apply::InitializeLumenDispatchTable;

pub fn once(function_symbols: &[FunctionSymbol]) {
    ONCE.call_once(|| {
        unsafe { InitializeLumenDispatchTable(function_symbols.as_ptr(), function_symbols.len()) };
    });
}

static ONCE: Once = Once::new();
