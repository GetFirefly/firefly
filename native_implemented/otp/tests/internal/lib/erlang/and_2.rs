#[path = "and_2/with_false_left.rs"]
mod with_false_left;
#[path = "and_2/with_true_left.rs"]
mod with_true_left;

test_stdout!(without_boolean_left_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
