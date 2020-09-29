#[path = "concatenate_2/with_empty_list.rs"]
mod with_empty_list;
#[path = "concatenate_2/with_non_empty_proper_list.rs"]
mod with_non_empty_proper_list;

test_stdout!(without_list_left_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
test_stdout!(
    with_improper_list_left_errors_badarg,
    "{caught, error, badarg}\n"
);
