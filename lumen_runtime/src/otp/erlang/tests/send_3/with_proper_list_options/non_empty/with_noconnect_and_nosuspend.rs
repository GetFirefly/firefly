use super::*;

mod with_tuple_destination;

fn options(process: &Process) -> Term {
    let noconnect = Term::str_to_atom("noconnect", DoNotCare).unwrap();
    let nosuspend = Term::str_to_atom("nosuspend", DoNotCare).unwrap();

    Term::slice_to_list(&[noconnect, nosuspend], process)
}
