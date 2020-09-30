test_stdout!(without_tuple_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
test_stdout!(
    with_tuple_without_integer_between_1_and_the_length_inclusive_errors_badarg,
    "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n"
);
test_stdout!(
    with_tuple_with_integer_between_1_and_the_length_inclusive_returns_element,
    "1\n1\n2\n"
);
