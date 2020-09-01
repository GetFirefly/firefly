test_stdout!(
    with_normal_exit_does_not_exit_parent_or_send_exit_message,
    "{in, child}\n{parent, alive}\n"
);
test_stdout_substrings!(
    without_normal_exit_does_not_exit_parent_or_send_exit_message,
    vec![
        "{in, child}",
        "exited with reason: abnormal",
        "{parent, alive}"
    ]
);
