use super::*;

#[test]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    super::in_same_thread_without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message(options);
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    crate::test::with_options_with_timer_in_same_thread_with_timeout_returns_false_after_timeout_message_was_sent(result, options);
}
