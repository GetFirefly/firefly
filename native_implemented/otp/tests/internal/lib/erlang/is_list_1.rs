test_stdout!(
    without_list_returns_false,
    "false\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\n"
);
test_stdout!(with_list_reutrns_true, "true\ntrue\ntrue\n");
