#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;
use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::string_to_integer::decimal_string_to_integer;
use crate::runtime::binary_to_string::binary_to_string;

#[native_implemented::function(erlang:binary_to_integer/1)]
pub fn result(process: &Process, binary: Term) -> Result<Term, NonNull<ErlangException>> {
    let string: String = binary_to_string(binary)?;

    decimal_string_to_integer(process, "binary", binary, &string).map_err(From::from)
}
