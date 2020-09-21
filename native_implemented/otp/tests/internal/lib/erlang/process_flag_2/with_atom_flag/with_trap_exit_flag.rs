// `without_boolean_value_errors_badarg` in unit tests
test_stdout!(with_boolean_returns_original_value_false, "false\n");
test_stdout!(
    with_true_value_then_boolean_value_returns_old_value_true,
    "false\ntrue\n"
);
test_stdout!(
    with_true_value_with_linked_and_does_not_exit_when_linked_process_exits_normal,
    "{trap_exit, true}\n{parent, alive, true}\n{child, exited, normal}\n"
);
test_stdout!(
    with_true_value_with_linked_receive_exit_message_and_does_not_exit_when_linked_process_does_not_exit_normal,
    "{trap_exit, true}\n{parent, alive, true}\n{child, exited, abnormal}\n{parent, sees, child, alive, false}\n"
);
test_stdout!(
    with_true_value_then_false_value_exits_when_linked_process_does_not_exit_normal,
    "{trap_exit, true}\n{trap_exit, false}\n{child, exited, abnormal}\n{parent, exited, abnormal}\n"
);
