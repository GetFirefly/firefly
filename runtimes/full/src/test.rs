#[cfg(test)]
pub mod r#loop;
#[cfg(test)]
pub mod process;

use std::sync::Once;

use liblumen_alloc::erts::apply::InitializeLumenDispatchTable;
use liblumen_core::symbols::FunctionSymbol;

use crate::scheduler;

pub fn once(function_symbols: &[FunctionSymbol]) {
    ONCE.call_once(|| {
        unsafe { InitializeLumenDispatchTable(function_symbols.as_ptr(), function_symbols.len()) };
        scheduler::set_unregistered_once();
    });
}

#[cfg(test)]
pub(crate) fn once_crate() {
    once(&[]);
}

static ONCE: Once = Once::new();
