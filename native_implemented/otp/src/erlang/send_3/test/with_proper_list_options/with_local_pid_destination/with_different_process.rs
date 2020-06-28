use super::*;

#[test]
fn without_locked_adds_process_message_to_mailbox_and_returns_ok() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                valid_options(arc_process.clone()),
            )
        },
        |(arc_process, message, options)| {
            let different_arc_process = test::process::child(&arc_process);
            let destination = different_arc_process.pid_term();

            prop_assert_eq!(
                result(&arc_process, destination, message, options),
                Ok(Atom::str_to_term("ok"))
            );

            prop_assert!(has_process_message(&different_arc_process, message));

            Ok(())
        },
    );
}

#[test]
fn with_locked_adds_process_message_to_mailbox_and_returns_ok() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                valid_options(arc_process.clone()),
            )
        },
        |(arc_process, message, options)| {
            let different_arc_process = test::process::child(&arc_process);
            let destination = different_arc_process.pid_term();

            let _different_process_heap_lock = different_arc_process.acquire_heap();

            assert_eq!(
                result(&arc_process, destination, message, options),
                Ok(Atom::str_to_term("ok"))
            );

            assert!(has_heap_message(&different_arc_process, message));

            Ok(())
        },
    );
}
