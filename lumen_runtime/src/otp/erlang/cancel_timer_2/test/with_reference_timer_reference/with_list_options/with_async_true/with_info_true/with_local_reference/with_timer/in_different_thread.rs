use super::*;

use crate::test::*;

#[test]
#[ignore]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer_in_different_thread(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half_and_wait(milliseconds, barrier);

        let timeout_message = different_timeout_message(timer_reference, process);

        assert!(!has_message(process, timeout_message));

        assert_eq!(
            native(process, timer_reference, options(process)),
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
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(false_cancel_timer_message));

        timeout_after_half_and_wait(milliseconds, barrier);

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(false_cancel_timer_message));
    });
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    with_timer_in_different_thread(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half_and_wait(milliseconds, barrier);
        timeout_after_half_and_wait(milliseconds, barrier);

        let timeout_message = different_timeout_message(timer_reference, process);

        assert_eq!(receive_message(process), Some(timeout_message));

        let cancel_timer_message = cancel_timer_message(timer_reference, false.into(), process);

        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(cancel_timer_message));

        // again
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(cancel_timer_message));
    });
}
