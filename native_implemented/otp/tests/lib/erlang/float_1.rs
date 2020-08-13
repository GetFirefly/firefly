// `without_number_errors_badarg` in unit tests
test_stdout!(
    with_integer_returns_float_with_same_value,
    "-1.0\n0.0\n1.0\n"
);
test_stdout!(with_float_returns_same_float, "-1.2\n0.3\n4.5\n");
