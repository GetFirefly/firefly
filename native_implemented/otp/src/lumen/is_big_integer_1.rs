use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(lumen:is_big_integer/1)]
pub fn result(term: Term) -> Term {
    match term.decode() {
        Ok(TypedTerm::BigInteger(_)) => true,
        _ => false,
    }
    .into()
}
