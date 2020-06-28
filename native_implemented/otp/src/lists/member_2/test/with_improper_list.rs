use super::*;

#[test]
fn without_found_errors_badarg() {
    with_process_arc(|arc_process| {
        let element = Atom::str_to_term("not_found");
        let not_element = Atom::str_to_term("not_element");
        let slice = &[not_element];
        let tail = Atom::str_to_term("tail");
        let list = arc_process.improper_list_from_slice(slice, tail).unwrap();

        assert_badarg!(
            result(element, list),
            format!("list ({}) is improper", list)
        );
    });
}

#[test]
fn with_found_returns_true() {
    with_process_arc(|arc_process| {
        let element = Atom::str_to_term("found");
        let slice = &[element];
        let tail = Atom::str_to_term("tail");
        let list = arc_process.improper_list_from_slice(slice, tail).unwrap();

        assert_eq!(result(element, list), Ok(true.into()));
    });
}
