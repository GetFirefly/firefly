test_stdout!(
    with_valid_arguments_when_run_exits_normal_and_parent_does_not_exit,
    "{child, sum, 3}\n{parent, alive, true}\n"
);
test_substrings!(
    without_valid_arguments_when_run_exits_and_parent_exits,
    vec!["{parent, exited, function_clause}"],
    vec![
        "Process (#PID<0.3.0>) exited abnormally.",
        "function_clause"
    ]
);
