use super::*;

mod with_info_false;
mod with_info_true;
mod without_info;

fn options(process: &Process) -> Term {
    process
        .cons(async_option(true, process), Term::NIL)
        .unwrap()
}

fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message(
    options: fn(&Process) -> Term,
) {
    with_timer_in_same_thread(|milliseconds, message, timer_reference, process| {
        let start_time_in_milliseconds = freeze_timeout();
        freeze_at_timeout(start_time_in_milliseconds + milliseconds / 2 + 1);

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );

        let received_message = receive_message(process).unwrap();

        let received_tuple_result: core::result::Result<Boxed<Tuple>, _> =
            received_message.try_into();

        assert!(received_tuple_result.is_ok());

        let received_tuple = received_tuple_result.unwrap();

        assert_eq!(received_tuple[0], Atom::str_to_term("cancel_timer"));
        assert_eq!(received_tuple[1], timer_reference);

        let milliseconds_remaining = received_tuple[2];

        assert!(milliseconds_remaining.is_integer());
        assert!(process.integer(0).unwrap() < milliseconds_remaining);
        assert!(milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        let false_cancel_timer_message =
            cancel_timer_message(timer_reference, false.into(), process);

        // again before timeout
        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(false_cancel_timer_message));

        freeze_at_timeout(start_time_in_milliseconds + milliseconds + 1);

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(false_cancel_timer_message));
    })
}

fn with_timeout_returns_ok_after_timeout_message_was_sent(options: fn(&Process) -> Term) {
    with_timer_in_same_thread(|milliseconds, message, timer_reference, process| {
        let start_time_in_milliseconds = freeze_timeout();
        freeze_at_timeout(start_time_in_milliseconds + milliseconds + 1);

        let timeout_message = timeout_message(timer_reference, message, process);

        assert_eq!(receive_message(process), Some(timeout_message));

        let cancel_timer_message = cancel_timer_message(timer_reference, false.into(), process);

        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(cancel_timer_message));

        // again
        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(cancel_timer_message));
    })
}
