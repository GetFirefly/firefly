use super::*;

mod with_tuple_destination;

fn options(process: &Process) -> Term {
    process.cons(Atom::str_to_term("nosuspend"), Term::NIL)
}
