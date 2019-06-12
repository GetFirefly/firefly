use super::*;

mod with_different_process;

#[test]
fn with_same_process_adds_process_message_to_mailbox_and_returns_message() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (Just(arc_process.clone()), strategy::term(arc_process))
            }),
            |(arc_process, message)| {
                let name = registered_name();

                prop_assert_eq!(
                    erlang::register_2(name, arc_process.pid, arc_process.clone()),
                    Ok(true.into()),
                    "Cannot register process ({:?}) pid ({:?}) with name ({:?})",
                    arc_process,
                    arc_process.pid,
                    name
                );

                let destination = Term::slice_to_tuple(&[name, erlang::node_0()], &arc_process);

                prop_assert_eq!(
                    erlang::send_2(destination, message, &arc_process),
                    Ok(message)
                );

                prop_assert!(has_process_message(&arc_process, message));

                Ok(())
            },
        )
        .unwrap();
}
