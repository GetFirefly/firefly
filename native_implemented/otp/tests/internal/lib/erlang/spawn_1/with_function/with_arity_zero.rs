test_stdout!(
    without_environment_runs_function_in_child_process,
    "no_environment\nnormal\n"
);
test_stdout!(
    with_environment_runs_function_in_child_process,
    "environment\nnormal\n"
);
