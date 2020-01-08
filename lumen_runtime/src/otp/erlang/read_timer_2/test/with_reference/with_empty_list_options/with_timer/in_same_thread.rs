use super::*;

use std::thread;
use std::time::Duration;

use crate::test::with_timer_in_same_thread;

#[test]
#[ignore]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer_in_same_thread(|milliseconds, message, timer_reference, process| {
        let half_milliseconds = milliseconds / 2;

        thread::sleep(Duration::from_millis(half_milliseconds + 1));
        timer::timeout();

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        let first_milliseconds_remaining =
            native(process, timer_reference, OPTIONS).expect("Timer could not be read");

        assert!(first_milliseconds_remaining.is_integer());
        // flaky
        assert!(process.integer(0).unwrap() < first_milliseconds_remaining);
        assert!(first_milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        // again before timeout
        let second_milliseconds_remaining =
            native(process, timer_reference, OPTIONS).expect("Timer could not be read");

        assert!(second_milliseconds_remaining.is_integer());
        assert!(second_milliseconds_remaining <= first_milliseconds_remaining);

        thread::sleep(Duration::from_millis(half_milliseconds + 1));
        timer::timeout();

        assert!(has_message(process, timeout_message));

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
