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

use crate::erlang::spawn_apply_3;

#[native_implemented_function(spawn/3)]
pub fn result(
    process: &Process,
    module: Term,
    function: Term,
    arguments: Term,
) -> exception::Result<Term> {
    spawn_apply_3::result(process, Default::default(), module, function, arguments)
}
