use super::*;

mod with_tuple_destination;

fn options(process: &Process) -> Term {
    let noconnect = Atom::str_to_term("noconnect");
    let nosuspend = Atom::str_to_term("nosuspend");

    process.list_from_slice(&[noconnect, nosuspend]).unwrap()
}
