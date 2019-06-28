use super::*;

mod with_tuple_destination;

fn options(process: &Process) -> Term {
    Term::cons(
        Term::str_to_atom("nosuspend", DoNotCare).unwrap(),
        Term::EMPTY_LIST,
        process,
    )
}
