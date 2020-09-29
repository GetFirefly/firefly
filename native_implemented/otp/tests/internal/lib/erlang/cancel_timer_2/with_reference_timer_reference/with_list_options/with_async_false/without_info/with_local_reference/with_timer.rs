test_stdout!(
    without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message,
    "no_message_at_midway\ntrue\ntrue\ntrue\nfalse\nno_message_after_cancel\nfalse\n"
);
test_stdout!(
    with_timeout_returns_false_after_timeout_message_was_sent,
    "message\nfalse\nfalse\n"
);
