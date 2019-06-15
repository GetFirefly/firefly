use super::*;

mod with_registered_name;

#[test]
fn without_registered_name_errors_badarg() {
    with_name_errors_badarg(|_| {
        Term::str_to_atom("without_registered_name_errors_badarg", DoNotCare).unwrap()
    });
}
