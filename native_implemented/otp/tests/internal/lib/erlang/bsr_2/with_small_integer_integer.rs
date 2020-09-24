test_stdout!(
    with_negative_with_overflow_shifts_left_and_returns_big_integer,
    "true\n36893488147419103232\n"
);
test_stdout!(
    with_negative_without_overflow_shifts_left_and_returns_small_integer,
    "true\n4\n"
);
test_stdout!(
    with_positive_without_underflow_returns_small_integer,
    "true\n1\n"
);
test_stdout!(with_positive_with_underflow_returns_zero, "0\n");
