use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::spawn_apply_3;
use crate::runtime::process::spawn::options::Options;

#[native_implemented::function(erlang:spawn_link/3)]
pub fn result(
    process: &Process,
    module: Term,
    function: Term,
    arguments: Term,
) -> exception::Result<Term> {
    let mut options: Options = Default::default();
    options.link = true;

    spawn_apply_3::result(process, options, module, function, arguments)
}
