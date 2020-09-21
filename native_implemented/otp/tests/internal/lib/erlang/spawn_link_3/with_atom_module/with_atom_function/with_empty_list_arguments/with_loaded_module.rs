#[path = "with_loaded_module/with_exported_function.rs"]
mod with_exported_function;

test_substrings!(
    without_exported_function_when_run_exits_undef_and_parent_exits,
    vec!["{parent, exited, undef}"],
    vec!["Process (#PID<0.3.0>) exited abnormally.", "undef"]
);
