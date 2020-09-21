#[path = "with_empty_list_options/with_arity_zero.rs"]
mod with_arity_zero;

test_stdout!(
    without_arity_zero_returns_pid_to_parent_and_child_process_exits_badarity,
    "badarity\n"
);
