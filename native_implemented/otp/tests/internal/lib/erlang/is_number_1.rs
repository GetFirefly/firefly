test_stdout!(
    without_number_returns_false,
    "false\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\n"
);
test_stdout!(with_number_returns_true, "true\ntrue\ntrue\n");
