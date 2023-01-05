#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;
use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::spawn_apply_3;

#[native_implemented::function(erlang:spawn/3)]
pub fn result(
    process: &Process,
    module: Term,
    function: Term,
    arguments: Term,
) -> Result<Term, NonNull<ErlangException>> {
    spawn_apply_3::result(process, Default::default(), module, function, arguments)
}
