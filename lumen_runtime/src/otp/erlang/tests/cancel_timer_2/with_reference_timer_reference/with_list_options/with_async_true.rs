use super::*;

mod with_info_false;
mod with_info_true;
mod without_info;

fn options(process: &ProcessControlBlock) -> Term {
    process
        .cons(async_option(true, process), Term::NIL)
        .unwrap()
}
