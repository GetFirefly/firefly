use super::*;

#[test]
fn without_found_returns_false() {
    with_process_arc(|arc_process| {
        let element = atom_unchecked("not_found");
        let slice = &[];
        let list = arc_process.list_from_slice(slice).unwrap();

        assert_eq!(native(element, list), Ok(false.into()));
    });
}

#[test]
fn with_found_returns_true() {
    with_process_arc(|arc_process| {
        let element = atom_unchecked("found");
        let slice = &[element];
        let list = arc_process.list_from_slice(slice).unwrap();

        assert_eq!(native(element, list), Ok(true.into()));
    });
}
