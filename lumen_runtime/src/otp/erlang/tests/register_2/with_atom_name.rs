use super::*;

mod without_registered_name;

#[test]
fn without_atom_name_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::pid_or_port(arc_process.clone()),
                |pid_or_port| {
                    let name = atom_unchecked("undefined");

                    prop_assert_eq!(
                        erlang::register_2(name, pid_or_port, arc_process.clone()),
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
        let registered_name = atom_unchecked("registered_name");

        assert_eq!(
            erlang::register_2(
                registered_name,
                unsafe { registered_process_arc.pid().as_term() },
                Arc::clone(&registered_process_arc)
            ),
            Ok(true.into())
        );

        let unregistered_process_arc = process::test(&registered_process_arc);

        assert_badarg!(erlang::register_2(
            registered_name,
            unregistered_process_arc.pid_term(),
            unregistered_process_arc
        ));
    });
}
