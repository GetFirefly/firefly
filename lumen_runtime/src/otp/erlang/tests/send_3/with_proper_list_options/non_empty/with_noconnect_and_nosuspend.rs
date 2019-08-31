use super::*;

mod with_tuple_destination;

fn options(process: &Process) -> Term {
    let noconnect = atom_unchecked("noconnect");
    let nosuspend = atom_unchecked("nosuspend");

    process.list_from_slice(&[noconnect, nosuspend]).unwrap()
}
