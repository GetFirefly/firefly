#[path = "with_non_empty_proper_list_arguments/with_loaded_module.rs"]
mod with_loaded_module;

test_stdout_substrings!(
    without_loaded_module_when_run_exits_undef_and_parent_does_not_exit,
    vec!["exited with reason: undef", "{parent, alive, true}"]
);
