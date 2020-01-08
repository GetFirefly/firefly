use super::*;

use std::thread;
use std::time::Duration;

use crate::test::{timeout_after_half, with_timer_in_same_thread};

#[test]
#[ignore]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer_in_same_thread(|milliseconds, message, timer_reference, process| {
        timeout_after_half(milliseconds);

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        let first_result = native(process, timer_reference, OPTIONS);

        assert!(first_result.is_ok());

        let milliseconds_remaining = first_result.unwrap();

        assert!(milliseconds_remaining.is_integer());
        assert!(process.integer(0).unwrap() < milliseconds_remaining);
        assert!(milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        // again before timeout
        assert_eq!(native(process, timer_reference, OPTIONS), Ok(false.into()));

        timeout_after_half(milliseconds);

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(native(process, timer_reference, OPTIONS), Ok(false.into()));
    })
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    with_timer_in_same_thread(|milliseconds, message, timer_reference, process| {
        thread::sleep(Duration::from_millis(milliseconds + 1));
        timer::timeout();

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(
            has_message(process, timeout_message),
            "Mailbox contains: {:?}",
            process.mailbox.lock().borrow()
        );

        assert_eq!(native(process, timer_reference, OPTIONS), Ok(false.into()));

        // again
        assert_eq!(native(process, timer_reference, OPTIONS), Ok(false.into()));
    })
}
