use super::*;

mod with_arity_2;

#[test]
fn without_arity_2_errors_badarg() {
    with_process(|process| {
        let destination = process.tuple_from_slice(&[]).unwrap();
        let message = atom_unchecked("message");

        assert_badarg!(erlang::send_2(destination, message, process))
    })
}
