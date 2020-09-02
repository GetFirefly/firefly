#[path = "with_non_empty_proper_list_arguments/with_loaded_module.rs"]
mod with_loaded_module;

test_substrings!(
    without_loaded_module_when_run_exits_undef_and_parent_does_not_exit,
    vec!["{parent, alive, true}"],
    vec!["Process (#PID<0.3.0>) exited abnormally.", "undef"]
);
