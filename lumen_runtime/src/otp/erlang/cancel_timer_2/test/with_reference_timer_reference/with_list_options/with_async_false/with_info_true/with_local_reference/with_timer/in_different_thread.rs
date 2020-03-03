use super::*;

#[test]
#[ignore]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    super::without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message(
        options,
    );
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    crate::test::with_timer_in_different_thread_with_timeout_returns_false_after_timeout_message_was_sent(native, options);
}
