use super::*;

mod with_async_false;
mod with_async_true;
mod with_invalid_option;
mod without_async;

fn async_option(value: bool, process: &Process) -> Term {
    option("async", value, process)
}

fn info_option(value: bool, process: &Process) -> Term {
    option("info", value, process)
}

fn option(key: &str, value: bool, process: &Process) -> Term {
    process
        .tuple_from_slice(&[atom_unchecked(key), value.into()])
        .unwrap()
}
