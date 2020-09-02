#[path = "with_loaded_module/with_exported_function.rs"]
mod with_exported_function;

test_substrings!(
    without_exported_function_when_run_exits_undef_and_sends_exit_message_to_parent,
    vec!["{child, exited, undef}"],
    vec!["Process exited abnormally.", "undef"]
);
