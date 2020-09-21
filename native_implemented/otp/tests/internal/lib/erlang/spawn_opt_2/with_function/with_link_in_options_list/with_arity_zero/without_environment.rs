test_stdout!(
    without_normal_exit_in_child_process_exits_linked_parent_process,
    "{parent, alive, false}\n"
);
test_stdout!(
    with_normal_exit_in_child_process_exits_linked_parent_process,
    "{child, exited, normal}\n{parent, alive, true}\n"
);
