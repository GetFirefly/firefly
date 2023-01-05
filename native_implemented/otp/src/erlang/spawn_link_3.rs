use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::spawn_apply_3;
use crate::runtime::process::spawn::options::Options;

#[native_implemented::function(erlang:spawn_link/3)]
pub fn result(
    process: &Process,
    module: Term,
    function: Term,
    arguments: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let mut options: Options = Default::default();
    options.link = true;

    spawn_apply_3::result(process, options, module, function, arguments)
}
