test_stdout!(without_number_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
test_stdout!(
    with_number_returns_non_negative,
    "18446744073709551616\n18446744073709551616\n1.2\n0.0\n3.4\n1\n0\n1\n"
);
