test_stdout!(
    with_arity_when_run_exits_normal_and_sends_exit_message_to_parent,
    "{child, ran}\n{child, exited, normal}\n{parent, alive, true}\n"
);
test_stdout_substrings!(
    without_arity_when_run_exits_undef_and_send_exit_message_to_parent,
    vec![
        "exited with reason: undef",
        "{child, exited, undef}",
        "{parent, alive, true}"
    ]
);
