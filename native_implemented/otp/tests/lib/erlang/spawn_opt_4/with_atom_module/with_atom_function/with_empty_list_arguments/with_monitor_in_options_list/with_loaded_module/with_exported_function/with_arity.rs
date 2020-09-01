test_stdout!(
    with_normal_exit_sends_exit_message_to_parent,
    "{in, child}\n{child, exited, normal}\n"
);
test_stdout_substrings!(
    without_normal_exit_sends_exit_message_to_parent,
    vec![
        "{in, child}",
        "exited with reason: abnormal",
        "{child, exited, abnormal}"
    ]
);
