use super::*;

#[test]
#[ignore]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer_in_same_thread_without_timeout_returns_ok_and_does_not_send_timeout_message(options);
}

#[test]
fn with_timeout_returns_ok_after_timeout_message_was_sent() {
    with_timer_in_same_thread_with_timeout_returns_ok_after_timeout_message_was_sent(options);
}
