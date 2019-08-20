use super::*;

#[test]
fn without_found_errors_badarg() {
    with_process_arc(|arc_process| {
        let element = atom_unchecked("not_found");
        let not_element = atom_unchecked("not_element");
        let slice = &[not_element];
        let tail = atom_unchecked("tail");
        let list = arc_process.improper_list_from_slice(slice, tail).unwrap();

        assert_eq!(native(element, list), Err(badarg!().into()));
    });
}

#[test]
fn with_found_returns_true() {
    with_process_arc(|arc_process| {
        let element = atom_unchecked("found");
        let slice = &[element];
        let tail = atom_unchecked("tail");
        let list = arc_process.improper_list_from_slice(slice, tail).unwrap();

        assert_eq!(native(element, list), Ok(true.into()));
    });
}
