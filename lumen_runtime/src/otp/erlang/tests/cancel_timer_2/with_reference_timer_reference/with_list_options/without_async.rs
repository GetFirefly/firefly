use super::*;

mod with_info_false;
mod with_info_true;
mod without_info;

fn options(_process: &ProcessControlBlock) -> Term {
    Term::NIL
}
