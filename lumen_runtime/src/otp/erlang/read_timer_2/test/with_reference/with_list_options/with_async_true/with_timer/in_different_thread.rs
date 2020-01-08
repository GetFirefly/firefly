use super::*;

use crate::test::*;

#[test]
#[ignore]
fn without_timeout_returns_milliseconds_remaining() {
    with_timer_in_different_thread(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half_and_wait(milliseconds, barrier);

        let message = Atom::str_to_term("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );

        let first_received_message = receive_message(process).unwrap();

        let first_received_tuple_result: core::result::Result<Boxed<Tuple>, _> =
            first_received_message.try_into();

        assert!(first_received_tuple_result.is_ok());

        let first_received_tuple = first_received_tuple_result.unwrap();

        assert_eq!(first_received_tuple[0], Atom::str_to_term("read_timer"));
        assert_eq!(first_received_tuple[1], timer_reference);

        let first_milliseconds_remaining = first_received_tuple[2];

        assert!(first_milliseconds_remaining.is_integer());
        assert!(process.integer(0).unwrap() < first_milliseconds_remaining);
        assert!(first_milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        // again before timeout
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );

        let second_received_message = receive_message(process).unwrap();

        let second_received_tuple_result: core::result::Result<Boxed<Tuple>, _> =
            second_received_message.try_into();

        assert!(second_received_tuple_result.is_ok());

        let second_received_tuple = second_received_tuple_result.unwrap();

        assert_eq!(second_received_tuple[0], Atom::str_to_term("read_timer"));
        assert_eq!(second_received_tuple[1], timer_reference);

        let second_milliseconds_remaining = second_received_tuple[2];

        assert!(second_milliseconds_remaining.is_integer());
        assert!(second_milliseconds_remaining <= first_milliseconds_remaining);

        timeout_after_half_and_wait(milliseconds, barrier);

        assert_eq!(receive_message(process), Some(timeout_message));

        let false_read_timer_message = read_timer_message(timer_reference, false.into(), process);

        // again after timeout
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(false_read_timer_message))
    });
}

#[test]
fn with_timeout_returns_ok_after_timeout_message_was_sent() {
    with_timer_in_different_thread(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half_and_wait(milliseconds, barrier);
        timeout_after_half_and_wait(milliseconds, barrier);

        let message = Atom::str_to_term("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        assert_eq!(receive_message(process), Some(timeout_message));

        let read_timer_message = read_timer_message(timer_reference, false.into(), process);

        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(read_timer_message));

        // again
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(read_timer_message));
    });
}
