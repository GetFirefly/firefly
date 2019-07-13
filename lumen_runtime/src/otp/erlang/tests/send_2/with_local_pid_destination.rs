use super::*;

mod with_different_process;

#[test]
fn without_process_returns_message() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::local(),
                    strategy::term(arc_process.clone()),
                ),
                |(destination, message)| {
                    prop_assert_eq!(
                        erlang::send_2(destination, message, &arc_process),
                        Ok(message)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_process_adds_process_message_to_mailbox_and_returns_message() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |message| {
                let destination = arc_process.pid_term();

                prop_assert_eq!(
                    erlang::send_2(destination, message, &arc_process.clone()),
                    Ok(message)
                );

                prop_assert!(has_process_message(&arc_process, message));

                Ok(())
            })
            .unwrap();
    });
}
