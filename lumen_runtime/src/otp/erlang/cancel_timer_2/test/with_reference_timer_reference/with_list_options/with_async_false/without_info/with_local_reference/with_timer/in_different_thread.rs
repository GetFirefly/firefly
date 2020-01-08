use super::*;

use crate::test::*;

#[test]
// flaky
#[ignore]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);

        let message = Atom::str_to_term("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        let milliseconds_remaining = native(process, timer_reference, options(process))
            .expect("Timer could not be cancelled");

        assert!(milliseconds_remaining.is_integer());
        assert!(process.integer(0).unwrap() < milliseconds_remaining);
        assert!(milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        // again before timeout
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(false.into())
        );

        timeout_after_half(milliseconds, barrier);

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(false.into())
        );
    });
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    with_timer(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);
        timeout_after_half(milliseconds, barrier);

        let message = Atom::str_to_term("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(
            has_message(process, timeout_message),
            "Mailbox does not contain {:?} and instead contains {:?}",
            timeout_message,
            process.mailbox.lock().borrow()
        );

        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(false.into())
        );

        // again
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(false.into())
        );
    });
}
