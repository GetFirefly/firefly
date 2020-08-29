#[path = "with_exported_function/with_arity.rs"]
mod with_arity;

test_stdout_substrings!(
    without_arity_when_run_exits_undef_and_parent_does_not_exit,
    vec![
        "exited with reason: undef",
        "{parent, alive, true}"
    ]
);
