use super::*;

use crate::test::*;

#[test]
#[ignore]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer_in_different_thread(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half_and_wait(milliseconds, barrier);

        let timeout_message = different_timeout_message(timer_reference, process);

        assert!(!has_message(process, timeout_message));

        let milliseconds_remaining =
            native(&process, timer_reference).expect("Timer could not be cancelled");

        assert!(milliseconds_remaining.is_integer());
        assert!(process.integer(0).unwrap() < milliseconds_remaining);
        assert!(milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        // again before timeout
        assert_eq!(native(&process, timer_reference), Ok(false.into()));

        timeout_after_half_and_wait(milliseconds, barrier);

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(native(&process, timer_reference), Ok(false.into()));
    });
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    with_timer_in_different_thread(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half_and_wait(milliseconds, barrier);
        timeout_after_half_and_wait(milliseconds, barrier);

        let timeout_message = different_timeout_message(timer_reference, process);

        assert_has_message!(process, timeout_message);

        assert_eq!(native(&process, timer_reference), Ok(false.into()));

        // again
        assert_eq!(native(&process, timer_reference), Ok(false.into()));
    });
}
