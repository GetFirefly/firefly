test_stdout!(without_tuple_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
test_stdout!(
    with_tuple_without_integer_between_1_and_the_length_plus_1_inclusive_errors_badarg,
    "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n"
);
test_stdout!(with_tuple_with_integer_between_1_and_the_length_plus_1_inclusive_returns_tuple_with_element, "{inserted_element}\n{inserted_element, 1}\n{1, inserted_element}\n{inserted_element, 1, 2}\n{1, inserted_element, 2}\n{1, 2, inserted_element}\n");
