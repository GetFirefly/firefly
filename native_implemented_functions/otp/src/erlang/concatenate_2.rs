// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

/// `++/2`
#[native_implemented_function(++/2)]
pub fn result(process: &Process, list: Term, term: Term) -> exception::Result<Term> {
    match list.decode()? {
        TypedTerm::Nil => Ok(term),
        TypedTerm::List(cons) => match cons
            .into_iter()
            .collect::<std::result::Result<Vec<Term>, _>>()
        {
            Ok(vec) => process
                .improper_list_from_slice(&vec, term)
                .map_err(|error| error.into()),
            Err(ImproperList { .. }) => Err(ImproperListError)
                .context(format!("list ({}) is improper", list))
                .map_err(From::from),
        },
        _ => Err(TypeError)
            .context(format!("list ({}) is not a list", list))
            .map_err(From::from),
    }
}
