test_stdout!(
    with_normal_exit_sends_exit_message_to_parent,
    "{in, child}\n{child, exited, normal}\n"
);
test_substrings!(
    without_normal_exit_sends_exit_message_to_parent,
    vec!["{in, child}", "{child, exited, abnormal}"],
    vec!["Process exited abnormally.", "abnormal"]
);
