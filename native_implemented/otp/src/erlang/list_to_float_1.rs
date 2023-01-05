#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::charlist_to_string::charlist_to_string;
use crate::erlang::string_to_float::string_to_float;

#[native_implemented::function(erlang:list_to_float/1)]
pub fn result(process: &Process, list: Term) -> Result<Term, NonNull<ErlangException>> {
    let string = charlist_to_string(list)?;

    string_to_float(process, "list", list, &string).map_err(From::from)
}
