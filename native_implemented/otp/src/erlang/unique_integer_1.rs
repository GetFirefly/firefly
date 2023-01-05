#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::unique_integer::{unique_integer, Options};

#[native_implemented::function(erlang:unique_integer/1)]
pub fn result(process: &Process, options: Term) -> Result<Term, NonNull<ErlangException>> {
    let options_options: Options = options.try_into()?;

    Ok(unique_integer(process, options_options))
}
