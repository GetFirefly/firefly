#[path = "with_exported_function/with_arity.rs"]
mod with_arity;

test_stdout_substrings!(
    without_arity_when_run_exits_undef_and_sends_exit_message_to_parent,
    vec!["exited with reason: undef"]
);
