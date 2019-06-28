use super::*;

#[test]
fn with_different_node_returns_nosuspend() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (Just(arc_process.clone()), strategy::term(arc_process))
            }),
            |(arc_process, message)| {
                let name = registered_name();

                prop_assert_eq!(
                    erlang::register_2(name, arc_process.pid, arc_process.clone()),
                    Ok(true.into())
                );

                let destination = Term::slice_to_tuple(
                    &[
                        name,
                        Term::str_to_atom("node@example.com", DoNotCare).unwrap(),
                    ],
                    &arc_process,
                );
                let options = options(&arc_process);

                prop_assert_eq!(
                    erlang::send_3(destination, message, options, &arc_process),
                    Ok(Term::str_to_atom("nosuspend", DoNotCare).unwrap())
                );

                Ok(())
            },
        )
        .unwrap();
}
