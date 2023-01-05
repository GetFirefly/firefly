use firefly_rt::term::Term;

#[native_implemented::function(erlang:is_atom/1)]
pub fn result(term: Term) -> Term {
    term.is_atom().into()
}
