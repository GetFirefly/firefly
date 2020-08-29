#[path = "with_empty_list_arguments/with_loaded_module.rs"]
mod with_loaded_module;

test_stdout_substrings!(
    without_loaded_module_when_run_exits_undef_and_sends_exit_message_to_parent,
    vec![
        "exited with reason: undef",
        "{child, exited, undef}",
        "{parent, alive, true}"
    ]
);
