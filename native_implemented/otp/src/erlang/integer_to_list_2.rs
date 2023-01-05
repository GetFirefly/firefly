#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::integer_to_string::base_integer_to_string;

#[native_implemented::function(erlang:integer_to_list/2)]
pub fn result(
    process: &Process,
    integer: Term,
    base: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let string = base_integer_to_string(base, integer)?;
    let charlist = process.charlist_from_str(&string);

    Ok(charlist)
}
