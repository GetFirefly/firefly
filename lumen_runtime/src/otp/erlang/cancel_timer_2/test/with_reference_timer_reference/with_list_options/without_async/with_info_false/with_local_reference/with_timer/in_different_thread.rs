use super::*;

use crate::test::*;

#[test]
#[ignore]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer_in_different_thread(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half_and_wait(milliseconds, barrier);

        let message = Atom::str_to_term("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );

        // again before timeout
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );

        timeout_after_half_and_wait(milliseconds, barrier);

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
    });
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    with_timer_in_different_thread(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half_and_wait(milliseconds, barrier);
        timeout_after_half_and_wait(milliseconds, barrier);

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
            Ok(Atom::str_to_term("ok"))
        );

        // again
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
    });
}
