#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::integer_to_string::decimal_integer_to_string;

#[native_implemented::function(erlang:integer_to_binary/1)]
pub fn result(process: &Process, integer: Term) -> Result<Term, NonNull<ErlangException>> {
    let string = decimal_integer_to_string(integer)?;
    let binary = process.binary_from_str(&string);

    Ok(binary)
}
