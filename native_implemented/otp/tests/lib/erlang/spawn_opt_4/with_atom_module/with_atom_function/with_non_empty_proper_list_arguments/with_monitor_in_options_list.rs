#[path = "with_monitor_in_options_list/with_loaded_module.rs"]
mod with_loaded_module;

test_substrings!(
    without_loaded_module_when_run_exits_undef_and_sends_exit_message_to_parent,
    vec!["{child, exited, undef}"],
    vec!["Process exited abnormally.", "undef"]
);
