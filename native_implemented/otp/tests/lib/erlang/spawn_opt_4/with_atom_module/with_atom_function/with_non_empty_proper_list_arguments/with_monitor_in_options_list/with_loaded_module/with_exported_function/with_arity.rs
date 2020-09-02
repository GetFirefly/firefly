test_stdout!(
    with_normal_exit_sends_exit_message_to_parent,
    "{in, child, 1, 2}\n{child, exited, normal}\n"
);
test_substrings!(
    without_normal_exit_sends_exit_message_to_parent,
    vec!["{in, child, 1, 2}", "{child, exited, abnormal}"],
    vec!["Process exited abnormally.", "abnormal"]
);
