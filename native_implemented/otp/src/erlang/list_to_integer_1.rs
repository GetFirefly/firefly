#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::list_to_string::list_to_string;
use crate::erlang::string_to_integer::decimal_string_to_integer;

#[native_implemented::function(erlang:list_to_integer/1)]
pub fn result(process: &Process, list: Term) -> Result<Term, NonNull<ErlangException>> {
    let string: String = list_to_string(list)?;

    decimal_string_to_integer(process, "list", list, &string).map_err(From::from)
}
