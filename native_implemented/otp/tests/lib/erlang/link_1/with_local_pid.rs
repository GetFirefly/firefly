test_stdout!(
    with_self_returns_true_but_does_not_create_link,
    "true\ntrue\n"
);
// `with_non_existent_pid_errors_noproc` in unit tests
test_stdout!(with_existing_unlinked_pid_links_to_process, "true\ntrue\n");
test_stdout!(with_existing_linked_pid_returns_true, "true\ntrue\n");
test_stdout!(
    when_a_linked_process_exits_normal_the_process_does_not_exit,
    "true\n{child, exited, normal}\n"
);
test_stdout!(
    when_a_linked_process_does_not_exit_normal_the_process_exits_too,
    "true\n{parent, exited, abnormal}\n"
);
test_stdout!(
    when_the_process_does_not_exit_normal_linked_processes_exit_too,
    "true\n{parent, exited, abnormal}\n{child, exited, abnormal}\n"
);
