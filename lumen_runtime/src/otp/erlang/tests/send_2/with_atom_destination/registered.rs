use super::*;

mod with_different_process;

#[test]
fn with_same_process_address_process_message_to_mailbox_and_returns_message() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (Just(arc_process.clone()), strategy::term(arc_process))
            }),
            |(arc_process, message)| {
                let destination = registered_name();

                prop_assert_eq!(
                    erlang::register_2(destination, arc_process.pid, arc_process.clone()),
                    Ok(true.into())
                );

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
