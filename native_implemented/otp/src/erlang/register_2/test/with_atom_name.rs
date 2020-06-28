use super::*;

mod without_registered_name;

#[test]
fn without_atom_name_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::pid_or_port(arc_process.clone()),
            )
        },
        |(arc_process, pid_or_port)| {
            let name = Atom::str_to_term("undefined");

            prop_assert_badarg!(
                result(arc_process.clone(), name, pid_or_port),
                "undefined is not an allowed registered name"
            );

            Ok(())
        },
    );
}

#[test]
fn with_registered_name_errors_badarg() {
    with_process_arc(|registered_process_arc| {
        let registered_name = Atom::str_to_term("registered_name");

        assert_eq!(
            result(
                Arc::clone(&registered_process_arc),
                registered_name,
                registered_process_arc.pid().into()
            ),
            Ok(true.into())
        );

        let unregistered_process_arc = test::process::child(&registered_process_arc);

        assert_badarg!(
            result(
                unregistered_process_arc.clone(),
                registered_name,
                unregistered_process_arc.pid_term(),
            ),
            format!(
                "{} could not be registered as {}.  It may already be registered.",
                unregistered_process_arc.pid_term(),
                registered_name
            )
        );
    });
}
