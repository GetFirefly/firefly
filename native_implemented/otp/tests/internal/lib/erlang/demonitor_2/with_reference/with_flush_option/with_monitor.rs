test_stdout!(returns_true, "true\n");
test_stdout!(
    flushes_existing_message_and_returns_true,
    "true\ntrue\nfalse\n"
);
test_stdout!(prevents_future_messages, "false\ntrue\nfalse\n");
