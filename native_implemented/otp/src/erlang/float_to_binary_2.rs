#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::float_to_string::{float_to_string, Options};

#[native_implemented::function(float_to_binary/2)]
pub fn result(process: &Process, float: Term, options: Term) -> exception::Result<Term> {
    let options_options: Options = options.try_into()?;

    float_to_string(float, options_options)
        .map_err(|error| error.into())
        .and_then(|string| {
            process
                .binary_from_str(&string)
                .map_err(|alloc| alloc.into())
        })
}
