use super::*;

mod with_tuple_destination;

fn options(process: &Process) -> Term {
    process
        .cons(atom_unchecked("nosuspend"), Term::NIL)
        .unwrap()
}
