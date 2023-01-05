#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::float_to_string::float_to_string;

#[native_implemented::function(erlang:float_to_list/1)]
pub fn result(process: &Process, float: Term) -> Result<Term, NonNull<ErlangException>> {
    float_to_string(float, Default::default())
        .map_err(|error| error.into())
        .map(|string| process.charlist_from_str(&string))
}
