use super::*;

#[test]
fn without_locked_adds_process_message_to_mailbox_and_returns_ok() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(message, options)| {
                    let different_arc_process = process::test(&arc_process);
                    let name = registered_name();

                    prop_assert_eq!(
                        erlang::register_2::native(
                            arc_process.clone(),
                            name,
                            different_arc_process.pid_term(),
                        ),
                        Ok(true.into())
                    );

                    let destination = arc_process
                        .tuple_from_slice(&[name, erlang::node_0::native()])
                        .unwrap();

                    prop_assert_eq!(
                        native(&arc_process, destination, message, options),
                        Ok(Atom::str_to_term("ok"))
                    );

                    prop_assert!(has_process_message(&different_arc_process, message));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_locked_adds_heap_message_to_mailbox_and_returns_ok() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(message, options)| {
                    let different_arc_process = process::test(&arc_process);
                    let name = registered_name();

                    prop_assert_eq!(
                        erlang::register_2::native(
                            arc_process.clone(),
                            name,
                            different_arc_process.pid_term(),
                        ),
                        Ok(true.into())
                    );

                    let _different_process_heap_lock = different_arc_process.acquire_heap();

                    let destination = arc_process
                        .tuple_from_slice(&[name, erlang::node_0::native()])
                        .unwrap();

                    prop_assert_eq!(
                        native(&arc_process, destination, message, options),
                        Ok(Atom::str_to_term("ok"))
                    );

                    prop_assert!(has_heap_message(&different_arc_process, message));

                    Ok(())
                },
            )
            .unwrap();
    });
}
