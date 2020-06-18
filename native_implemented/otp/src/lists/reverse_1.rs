#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::lists::reverse_2;

#[native_implemented::function(lists:reverse/1)]
fn result(process: &Process, list: Term) -> exception::Result<Term> {
    reverse_2::result(process, list, Term::NIL)
}
