test_stdout!(
    without_map_returns_false,
    "false\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\n"
);
test_stdout!(with_map_returns_true, "true\n");
