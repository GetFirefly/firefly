// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use crate::otp::erlang::list_to_string::list_to_string;

#[native_implemented_function(list_to_atom/1)]
pub fn native(process: &Process, string: Term) -> exception::Result<Term> {
    list_to_string(process, string).and_then(|s| match Atom::try_from_str(s) {
        Ok(atom) => Ok(atom.encode()?),
        Err(_) => Err(badarg!(process).into()),
    })
}
