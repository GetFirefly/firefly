test_stdout!(
    without_normal_exit_in_child_process_sends_exit_message_to_parent,
    "{child, exited, abnormal}\n"
);
test_stdout!(
    with_normal_exit_in_child_process_sends_exit_message_to_parent,
    "{child, exited, normal}\n"
);
