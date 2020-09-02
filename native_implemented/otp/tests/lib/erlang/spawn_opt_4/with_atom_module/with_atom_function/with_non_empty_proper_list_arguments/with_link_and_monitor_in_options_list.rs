#[path = "with_link_and_monitor_in_options_list/with_loaded_module.rs"]
mod with_loaded_module;

test_substrings!(
    without_loaded_module_when_run_exits_undef_and_parent_exits,
    vec!["{parent, exited, undef}"],
    vec!["Process exited abnormally.", "undef"]
);
