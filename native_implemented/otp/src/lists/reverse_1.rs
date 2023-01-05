#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::lists::reverse_2;

#[native_implemented::function(lists:reverse/1)]
fn result(process: &Process, list: Term) -> Result<Term, NonNull<ErlangException>> {
    reverse_2::result(process, list, Term::Nil)
}
