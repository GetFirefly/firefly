#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::spawn_apply_3;
use crate::runtime::process::spawn::options::Options;

#[native_implemented::function(spawn_opt/4)]
pub fn result(
    process: &Process,
    module: Term,
    function: Term,
    arguments: Term,
    options: Term,
) -> exception::Result<Term> {
    let options_options: Options = options.try_into()?;

    spawn_apply_3::result(process, options_options, module, function, arguments)
}
