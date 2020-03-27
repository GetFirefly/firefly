use super::*;

#[test]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    super::without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message(
        options,
    );
}

#[test]
fn with_timeout_returns_ok_after_timeout_message_was_sent() {
    super::with_timeout_returns_ok_after_timeout_message_was_sent(options);
}
