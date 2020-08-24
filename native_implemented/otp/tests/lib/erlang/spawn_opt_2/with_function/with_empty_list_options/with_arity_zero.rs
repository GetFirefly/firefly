test_stdout!(
    without_environment_runs_function_in_child_process,
    "from_fun\n"
);
test_stdout!(
    with_environment_runs_function_in_child_process,
    "from_environment\n"
);
