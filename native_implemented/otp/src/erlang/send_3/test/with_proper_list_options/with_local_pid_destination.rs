use super::*;

mod with_different_process;

#[test]
fn without_process_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                valid_options(arc_process.clone()),
            )
        },
        |(arc_process, message, options)| {
            let destination = Pid::next_term();

            prop_assert_eq!(
                result(&arc_process, destination, message, options),
                Ok(Atom::str_to_term("ok"))
            );

            Ok(())
        },
    );
}

#[test]
fn with_same_process_adds_process_message_to_mailbox_and_returns_ok() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                valid_options(arc_process.clone()),
            )
        },
        |(arc_process, message, options)| {
            let destination = arc_process.pid_term();

            prop_assert_eq!(
                result(&arc_process, destination, message, options),
                Ok(Atom::str_to_term("ok"))
            );

            prop_assert!(has_process_message(&arc_process, message));

            Ok(())
        },
    );
}
