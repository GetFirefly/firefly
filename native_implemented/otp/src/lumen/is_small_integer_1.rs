use firefly_rt::*;
use firefly_rt::term::Term;

#[native_implemented::function(lumen:is_small_integer/1)]
pub fn result(term: Term) -> Term {
    term.is_smallint().into()
}
