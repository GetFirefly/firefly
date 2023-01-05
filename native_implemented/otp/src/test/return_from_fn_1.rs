use firefly_rt::term::Term;

#[native_implemented::function(test:return_from_fn/1)]
fn result(argument: Term) -> Term {
    argument
}
