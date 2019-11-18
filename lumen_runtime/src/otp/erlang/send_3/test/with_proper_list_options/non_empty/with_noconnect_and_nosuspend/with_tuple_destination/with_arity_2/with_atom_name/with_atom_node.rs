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
                    erlang::register_2::native(arc_process.clone(), name, arc_process.pid_term()),
                    Ok(true.into())
                );

                let destination = arc_process
                    .tuple_from_slice(&[name, Atom::str_to_term("node@example.com")])
                    .unwrap();
                let options = options(&arc_process);

                prop_assert_eq!(
                    native(&arc_process, destination, message, options),
                    Ok(Atom::str_to_term("noconnect"))
                );

                Ok(())
            },
        )
        .unwrap();
}
