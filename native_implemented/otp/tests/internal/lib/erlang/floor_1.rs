test_stdout!(without_number_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
test_stdout!(with_integer_returns_integer, "true\ntrue\ntrue\ntrue\n");
test_stdout!(
    with_float_rounds_down_to_previous_integer,
    "{-1.2, -2}\n{-1.0, -1}\n{-0.3, -1}\n{0.0, 0}\n{0.4, 0}\n{1.0, 1}\n{1.5, 1}\n"
);
