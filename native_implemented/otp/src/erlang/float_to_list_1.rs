#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::float_to_string::float_to_string;

#[native_implemented::function(erlang:float_to_list/1)]
pub fn result(process: &Process, float: Term) -> exception::Result<Term> {
    float_to_string(float, Default::default())
        .map_err(|error| error.into())
        .and_then(|string| {
            process
                .charlist_from_str(&string)
                .map_err(|alloc| alloc.into())
        })
}
