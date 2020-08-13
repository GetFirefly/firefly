test_stdout!(returns_true, "true\ntrue\ntrue\n");
test_stdout!(prevents_future_messages, "true\ntrue\n");
// `does_not_flush_existing_message` in unit tests
