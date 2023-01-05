#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;
use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::spawn_apply_1;
use crate::runtime::process::spawn::options::Options;

#[native_implemented::function(erlang:spawn_link/1)]
pub fn result(process: &Process, function: Term) -> Result<Term, NonNull<ErlangException>> {
    spawn_apply_1::result(
        process,
        function,
        Options {
            link: true,
            ..Default::default()
        },
    )
}
