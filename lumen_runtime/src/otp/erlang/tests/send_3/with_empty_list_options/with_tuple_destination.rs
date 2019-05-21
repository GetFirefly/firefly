use super::*;

mod with_arity_2;

#[test]
fn without_arity_2_errors_badarg() {
    with_process(|process| {
        let destination = Term::slice_to_tuple(&[], process);
        let message = Term::str_to_atom("message", DoNotCare).unwrap();
        let options = Term::EMPTY_LIST;

        assert_badarg!(erlang::send_3(destination, message, options, process))
    })
}
