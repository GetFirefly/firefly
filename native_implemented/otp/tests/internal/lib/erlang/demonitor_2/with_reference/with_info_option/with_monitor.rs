test_stdout!(returns_true, "true\ntrue\ntrue\n");
test_stdout!(does_not_flush_existing_message, "true\ntrue\ntrue\n");
test_stdout!(prevents_future_messages, "false\ntrue\nfalse\n");
