use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(lumen:is_small_integer/1)]
pub fn result(term: Term) -> Term {
    term.is_smallint().into()
}
