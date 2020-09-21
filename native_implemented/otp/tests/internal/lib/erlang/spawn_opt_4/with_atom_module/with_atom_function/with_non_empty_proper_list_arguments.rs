#[path = "with_non_empty_proper_list_arguments/with_empty_list_options.rs"]
mod with_empty_list_options;
#[path = "with_non_empty_proper_list_arguments/with_link_and_monitor_in_options_list.rs"]
mod with_link_and_monitor_in_options_list;
#[path = "with_non_empty_proper_list_arguments/with_link_in_options_list.rs"]
mod with_link_in_options_list;
#[path = "with_non_empty_proper_list_arguments/with_monitor_in_options_list.rs"]
mod with_monitor_in_options_list;

test_stdout!(
    without_proper_list_options_errors_badarg,
    "{caught, error, badarg}\n"
);
