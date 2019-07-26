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
                    let destination = different_arc_process.pid_term();

                    prop_assert_eq!(
                        erlang::send_3(destination, message, options, &arc_process),
                        Ok(atom_unchecked("ok"))
                    );

                    prop_assert!(has_process_message(&different_arc_process, message));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_locked_adds_process_message_to_mailbox_and_returns_ok() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(message, options)| {
                    let different_arc_process = process::test(&arc_process);
                    let destination = different_arc_process.pid_term();

                    let _different_process_heap_lock = different_arc_process.acquire_heap();

                    assert_eq!(
                        erlang::send_3(destination, message, options, &arc_process),
                        Ok(atom_unchecked("ok"))
                    );

                    assert!(has_heap_message(&different_arc_process, message));

                    Ok(())
                },
            )
            .unwrap();
    });
}
