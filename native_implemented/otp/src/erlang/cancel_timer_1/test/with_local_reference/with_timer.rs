use super::*;

#[test]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer_in_same_thread(|milliseconds, message, timer_reference, process| {
        let start_time_in_milliseconds = freeze_timeout();
        freeze_at_timeout(start_time_in_milliseconds + milliseconds / 2 + 1);

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        let first_result = result(process, timer_reference);

        assert!(first_result.is_ok());

        let milliseconds_remaining = first_result.unwrap();

        assert!(milliseconds_remaining.is_integer());
        assert!(process.integer(0).unwrap() < milliseconds_remaining);
        assert!(milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        // again before timeout
        assert_eq!(result(process, timer_reference), Ok(false.into()));

        freeze_at_timeout(start_time_in_milliseconds + milliseconds + 1);

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(result(process, timer_reference), Ok(false.into()));
    })
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    crate::test::with_timer_in_same_thread_with_timeout_returns_false_after_timeout_message_was_sent(result);
}
