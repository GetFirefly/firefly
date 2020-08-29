#[path = "with_empty_list_options/with_arity_zero.rs"]
mod with_arity_zero;

test_stdout_substrings!(
    without_arity_zero_returns_pid_to_parent_and_child_process_exits_undef,
    vec!["exited with reason: undef", "{parent, alive}"]
);
