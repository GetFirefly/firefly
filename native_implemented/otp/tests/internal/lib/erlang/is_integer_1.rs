test_stdout!(
    without_integer_returns_false,
    "false\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\n"
);
test_stdout!(with_integer_returns_true, "true\ntrue\n");
