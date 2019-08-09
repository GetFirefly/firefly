use super::*;

mod with_tuple_destination;

fn options(process: &ProcessControlBlock) -> Term {
    process
        .cons(atom_unchecked("nosuspend"), Term::NIL)
        .unwrap()
}
