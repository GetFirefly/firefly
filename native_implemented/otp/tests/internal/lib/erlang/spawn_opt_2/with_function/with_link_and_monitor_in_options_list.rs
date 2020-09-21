#[path = "with_link_and_monitor_in_options_list/with_arity_zero.rs"]
mod with_arity_zero;

test_stdout!(
    without_arity_zero_returns_pid_to_parent_and_child_process_exits_badarity_and_exits_parent,
    "{parent, badarity}\n"
);
