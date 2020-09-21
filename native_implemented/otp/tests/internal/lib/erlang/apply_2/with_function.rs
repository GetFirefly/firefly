#[path = "with_function/with_anonymous.rs"]
pub mod with_anonymous;
#[path = "with_function/with_export.rs"]
pub mod with_export;

test_stdout!(without_list_arguments_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
test_stdout!(
    with_list_without_proper_arguments_errors_badarg,
    "{caught, error, badarg}\n"
);
