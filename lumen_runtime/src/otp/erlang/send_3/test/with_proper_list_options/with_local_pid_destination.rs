use super::*;

mod with_different_process;

#[test]
fn without_process_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(message, options)| {
                    let destination = next_pid();

                    prop_assert_eq!(
                        native(&arc_process, destination, message, options),
                        Ok(atom_unchecked("ok"))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_process_adds_process_message_to_mailbox_and_returns_ok() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(message, options)| {
                    let destination = arc_process.pid_term();

                    prop_assert_eq!(
                        native(&arc_process, destination, message, options),
                        Ok(atom_unchecked("ok"))
                    );

                    prop_assert!(has_process_message(&arc_process, message));

                    Ok(())
                },
            )
            .unwrap();
    });
}
