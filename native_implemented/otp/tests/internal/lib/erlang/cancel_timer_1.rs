#[path = "cancel_timer_1/with_local_reference.rs"]
pub mod with_local_reference;

test_stdout!(without_reference_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
