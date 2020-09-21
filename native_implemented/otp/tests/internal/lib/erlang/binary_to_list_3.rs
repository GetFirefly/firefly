// `without_binary_errors_badarg` in unit tests
// `with_binary_without_integer_start_errors_badarg` in unit tests
// `with_binary_with_positive_integer_start_without_integer_stop_errors_badarg` in unit tests

test_stdout!(
    with_binary_with_start_less_than_or_equal_to_stop_returns_list_of_bytes,
    "[0]\n[0, 1]\n[0, 1, 2]\n[1]\n[1, 2]\n[2]\n"
);

// `with_binary_with_start_greater_than_stop_errors_badarg` in unit tests
