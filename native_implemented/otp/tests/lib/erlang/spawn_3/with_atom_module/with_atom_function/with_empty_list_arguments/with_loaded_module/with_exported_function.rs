test_stdout_substrings!(
    with_arity_when_run_exits_normal_and_parent_does_not_exit,
    vec![
        "{child, ran}",
        "{child, exited, normal}",
        "{parent, alive, true}"
    ]
);
test_stdout_substrings!(
    without_arity_when_run_exits_undef_and_parent_does_not_exit,
    vec![
        "exited with reason: undef",
        "Source: erlang:+/0 is not exported",
        "{parent, alive, true}"
    ]
);
