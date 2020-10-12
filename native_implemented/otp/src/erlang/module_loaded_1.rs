use liblumen_alloc::erts::apply::module_loaded;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(erlang:module_loaded/1)]
pub fn result(module: Term) -> exception::Result<Term> {
    let module_atom = term_try_into_atom!(module)?;

    Ok(module_loaded(module_atom).into())
}
