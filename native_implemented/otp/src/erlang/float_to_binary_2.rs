#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::float_to_string::{float_to_string, Options};

#[native_implemented::function(erlang:float_to_binary/2)]
pub fn result(
    process: &Process,
    float: Term,
    options: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let options_options: Options = options.try_into()?;

    float_to_string(float, options_options)
        .map_err(|error| error.into())
        .map(|string| process.binary_from_str(&string))
}
