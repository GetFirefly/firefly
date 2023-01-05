use super::*;

mod with_tuple_destination;

fn options(process: &Process) -> Term {
    let noconnect = Atom::str_to_term("noconnect").into();
    let nosuspend = Atom::str_to_term("nosuspend").into();

    process.list_from_slice(&[noconnect, nosuspend]).unwrap()
}
