#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::float_to_string::float_to_string;

#[native_implemented::function(float_to_binary/1)]
pub fn result(process: &Process, float: Term) -> exception::Result<Term> {
    float_to_string(float, Default::default())
        .map_err(|error| error.into())
        .and_then(|string| {
            process
                .binary_from_str(&string)
                .map_err(|alloc| alloc.into())
        })
}
