#[path = "with_function/with_arity_zero.rs"]
pub mod with_arity_zero;

test_stdout!(without_arity_zero_returns_pid_to_parent_and_child_process_exits_badarity_and_sends_exit_message_to_parent, "{child, exited, badarity}\n");
