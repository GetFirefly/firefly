test_stdout!(
    with_exits_normal_arent_does_not_exit,
    "{in, child, 1, 2}\n{parent, alive, true}\n"
);
test_substrings!(
    without_exits_normal_arent_does_not_exit,
    vec!["{in, child, 1, 2}", "{parent, exited, abnormal}"],
    vec!["Process (#PID<0.3.0>) exited abnormally.", "abnormal"]
);
