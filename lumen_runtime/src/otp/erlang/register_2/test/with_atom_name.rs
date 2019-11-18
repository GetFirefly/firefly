use super::*;

mod without_registered_name;

#[test]
fn without_atom_name_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::pid_or_port(arc_process.clone()),
                |pid_or_port| {
                    let name = Atom::str_to_term("undefined");

                    prop_assert_eq!(
                        native(arc_process.clone(), name, pid_or_port),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_registered_name_errors_badarg() {
    with_process_arc(|registered_process_arc| {
        let registered_name = Atom::str_to_term("registered_name");

        assert_eq!(
            native(
                Arc::clone(&registered_process_arc),
                registered_name,
                registered_process_arc.pid().into()
            ),
            Ok(true.into())
        );

        let unregistered_process_arc = process::test(&registered_process_arc);

        assert_badarg!(native(
            unregistered_process_arc.clone(),
            registered_name,
            unregistered_process_arc.pid_term(),
        ));
    });
}
