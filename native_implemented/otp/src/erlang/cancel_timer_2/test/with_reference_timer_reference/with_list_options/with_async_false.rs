use super::*;

mod with_info_false;
mod with_info_true;
mod without_info;

fn options(process: &Process) -> Term {
    process.cons(async_option(false, process), Term::NIL)
}
