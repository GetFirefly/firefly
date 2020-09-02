test_stdout!(
    with_valid_arguments_when_run_exits_normal_and_sends_exit_message_to_parent,
    "{child, sum, 3}\n{child, exited, normal}\n{parent, alive, true}\n"
);
test_substrings!(
    without_valid_arguments_when_run_exits_and_sends_exit_message_to_parent,
    vec!["{child, exited, function_clause}", "{parent, alive, true}"],
    vec!["Process exited abnormally.", "function_clause"]
);
