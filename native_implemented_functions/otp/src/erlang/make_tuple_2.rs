// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

use crate::runtime::context::*;

#[native_implemented_function(make_tuple/2)]
pub fn native(process: &Process, arity: Term, initial_value: Term) -> exception::Result<Term> {
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
