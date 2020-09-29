#[path = "with_reference/with_flush_and_info_options.rs"]
mod with_flush_and_info_options;
#[path = "with_reference/with_flush_option.rs"]
mod with_flush_option;
#[path = "with_reference/with_info_option.rs"]
mod with_info_option;
#[path = "with_reference/without_options.rs"]
mod without_options;

test_stdout!(without_proper_list_for_options_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
test_stdout!(
    with_unknown_option_errors_badarg,
    "{caught, error, badarg}\n"
);
