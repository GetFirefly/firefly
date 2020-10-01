test_stdout!(
    without_boolean_returns_false,
    "false\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\n"
);
test_stdout!(with_boolean_returns_true, "true\ntrue\n");
