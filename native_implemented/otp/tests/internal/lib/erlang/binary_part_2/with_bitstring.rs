#[path = "with_bitstring/with_tuple_with_arity_2.rs"]
pub mod with_tuple_with_arity_2;

test_stdout!(without_tuple_start_length_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
test_stdout!(
    with_tuple_without_arity_2_errors_badarg,
    "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n"
);
