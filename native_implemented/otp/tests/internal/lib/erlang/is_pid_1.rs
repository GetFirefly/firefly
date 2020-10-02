test_stdout!(
    without_pid_returns_false,
    "false\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\n"
);
test_stdout!(with_pid_returns_true, "true\n");
