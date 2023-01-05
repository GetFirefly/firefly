#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::spawn_apply_1;
use crate::runtime::process::spawn::options::Options;

#[native_implemented::function(erlang:spawn_opt/2)]
pub fn result(
    process: &Process,
    function: Term,
    options: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let options: Options = options.try_into()?;

    spawn_apply_1::result(process, function, options)
}
