use super::*;

mod with_async_false;
mod with_async_true;
mod with_invalid_option;
mod without_async;

fn async_option(value: bool, process: &Process) -> Term {
    option("async", value, process)
}

fn option(key: &str, value: bool, process: &Process) -> Term {
    Term::slice_to_tuple(
        &[Term::str_to_atom(key, DoNotCare).unwrap(), value.into()],
        process,
    )
}
