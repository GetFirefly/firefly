test_stdout!(
    with_timeout_returns_ok_after_timeout_message_was_sent,
    "message\nok\nno_message_after_first_cancel\nok\nno_message_after_second_cancel\n"
);
test_stdout!(
    without_timeout_returns_ok_and_does_not_send_timeout_message,
    "no_message_at_midway\nok\nno_message_after_first_cancel\nok\nno_message_after_second_cancel\n"
);
