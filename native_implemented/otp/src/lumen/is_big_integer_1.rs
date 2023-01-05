use firefly_rt::term::Term;

#[native_implemented::function(lumen:is_big_integer/1)]
pub fn result(term: Term) -> Term {
    match term {
        Term::BigInt(_) => true,
        _ => false,
    }
    .into()
}
