// TODO with_*_flag
// #[path = "system_flag_2/with_*_flag.rs"]
// mod with_*_flag;

test_stdout!(without_atom_flag_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
test_stdout!(
    without_supported_atom_flag_errors_badarg,
    "{caught, error, badarg}\n"
);
