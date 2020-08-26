#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::spawn_apply_1;
use crate::runtime::process::spawn::options::Options;

#[native_implemented::function(erlang:spawn_opt/2)]
pub fn result(process: &Process, function: Term, options: Term) -> exception::Result<Term> {
    let options: Options = options.try_into()?;

    spawn_apply_1::result(process, function, options)
}
