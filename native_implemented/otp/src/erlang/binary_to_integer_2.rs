#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::string_to_integer::base_string_to_integer;
use crate::runtime::binary_to_string::binary_to_string;

#[native_implemented::function(erlang:binary_to_integer/2)]
pub fn result(
    process: &Process,
    binary: Term,
    base: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let string: String = binary_to_string(binary)?;

    base_string_to_integer(process, base, "binary", binary, &string).map_err(From::from)
}
