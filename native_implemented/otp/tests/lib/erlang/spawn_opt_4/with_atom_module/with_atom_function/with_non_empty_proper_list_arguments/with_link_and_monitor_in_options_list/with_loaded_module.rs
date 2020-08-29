#[path = "with_loaded_module/with_exported_function.rs"]
mod with_exported_function;

test_stdout_substrings!(
    without_exported_function_when_run_exits_undef_and_parent_exits,
    vec!["exited with reason: undef", "{parent, exited, undef}"]
);
