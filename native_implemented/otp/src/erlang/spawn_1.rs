#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::spawn_apply_1;

#[native_implemented::function(erlang:spawn/1)]
pub fn result(process: &Process, function: Term) -> exception::Result<Term> {
    spawn_apply_1::result(process, function, Default::default())
}
