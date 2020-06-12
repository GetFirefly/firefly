use super::*;

mod with_different_process;

#[test]
fn with_same_process_adds_process_message_to_mailbox_and_returns_ok() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                valid_options(arc_process),
            )
        },
        |(arc_process, message, options)| {
            let destination = registered_name();

            prop_assert_eq!(
                erlang::register_2::result(
                    arc_process.clone(),
                    destination,
                    arc_process.pid_term(),
                ),
                Ok(true.into())
            );

            prop_assert_eq!(
                result(&arc_process, destination, message, options),
                Ok(Atom::str_to_term("ok"))
            );

            assert!(has_process_message(&arc_process, message));

            Ok(())
        },
    );
}
