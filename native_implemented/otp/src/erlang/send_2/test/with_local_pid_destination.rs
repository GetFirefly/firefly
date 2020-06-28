use super::*;

mod with_different_process;

#[test]
fn without_process_returns_message() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::pid::local(),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, destination, message)| {
            prop_assert_eq!(result(&arc_process, destination, message), Ok(message));

            Ok(())
        },
    );
}

#[test]
fn with_same_process_adds_process_message_to_mailbox_and_returns_message() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, message)| {
            let destination = arc_process.pid_term();

            prop_assert_eq!(result(&arc_process, destination, message), Ok(message));

            prop_assert!(has_process_message(&arc_process, message));

            Ok(())
        },
    );
}
