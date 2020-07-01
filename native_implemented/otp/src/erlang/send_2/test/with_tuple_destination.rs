use super::*;

mod with_arity_2;

#[test]
fn without_arity_2_errors_badarg() {
    with_process(|process| {
        let destination = process.tuple_from_slice(&[]).unwrap();
        let message = Atom::str_to_term("message");

        assert_badarg!(
            result(process, destination, message),
            format!("destination ({}) is a tuple, but not 2-arity", destination)
        )
    })
}
