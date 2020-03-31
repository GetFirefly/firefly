// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

use crate::erlang::spawn_apply_1;
use crate::runtime::process::spawn::options::Options;

#[native_implemented_function(spawn_opt/2)]
pub fn native(process: &Process, function: Term, options: Term) -> exception::Result<Term> {
    let options: Options = options.try_into()?;

    spawn_apply_1::native(process, options, function)
}
