test_stdout!(
    with_exits_normal_arent_does_not_exit,
    "{in, child, 1, 2}\n{parent, alive, true}\n"
);
test_stdout_substrings!(
    without_exits_normal_arent_does_not_exit,
    vec![
        "{in, child, 1, 2}",
        "exited with reason: abnormal",
        "{parent, exited, abnormal}"
    ]
);
