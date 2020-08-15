test_stdout!(with_atom_returns_true, "true\n");
test_stdout!(
    without_atom_returns_false,
    "false\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\nfalse\n"
);
