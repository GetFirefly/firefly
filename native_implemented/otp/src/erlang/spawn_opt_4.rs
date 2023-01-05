use std::convert::TryInto;
use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::spawn_apply_3;
use crate::runtime::process::spawn::options::Options;

#[native_implemented::function(erlang:spawn_opt/4)]
pub fn result(
    process: &Process,
    module: Term,
    function: Term,
    arguments: Term,
    options: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let options_options: Options = options.try_into()?;

    spawn_apply_3::result(process, options_options, module, function, arguments)
}
