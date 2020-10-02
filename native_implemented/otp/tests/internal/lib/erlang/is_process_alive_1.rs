#[path = "is_process_alive_1/with_self.rs"]
mod with_self;
#[path = "is_process_alive_1/without_self.rs"]
mod without_self;

test_stdout!(without_pid_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
