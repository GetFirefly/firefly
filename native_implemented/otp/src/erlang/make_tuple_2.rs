#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::runtime::context::*;

#[native_implemented::function(erlang:make_tuple/2)]
pub fn result(process: &Process, arity: Term, initial_value: Term) -> exception::Result<Term> {
    // arity by definition is only 0-225, so `u8`, but ...
    let arity_u8: u8 = term_try_into_arity(arity)?;
    // ... everything else uses `usize`, so cast it back up
    let arity_usize: usize = arity_u8 as usize;

    process
        .tuple_from_iter(
            std::iter::repeat(initial_value).take(arity_usize),
            arity_usize,
        )
        .map_err(|alloc| alloc.into())
}
