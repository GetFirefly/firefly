use super::*;

mod without_registered_name;

#[test]
fn with_undefined_atom_errors_badarg() {
    with_name_errors_badarg(|_| Term::str_to_atom("undefined", DoNotCare).unwrap())
}

#[test]
fn with_registered_name_errors_badarg() {
    with_process_arc(|registered_process_arc| {
        let registered_name = Term::str_to_atom("registered_name", DoNotCare).unwrap();

        assert_eq!(
            erlang::register_2(
                registered_name,
                registered_process_arc.pid,
                Arc::clone(&registered_process_arc)
            ),
            Ok(true.into())
        );

        let unregistered_process_arc = process::local::test(&registered_process_arc);

        assert_badarg!(erlang::register_2(
            registered_name,
            unregistered_process_arc.pid,
            unregistered_process_arc
        ));
    });
}
