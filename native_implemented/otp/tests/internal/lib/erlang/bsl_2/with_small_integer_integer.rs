test_stdout!(with_negative_without_underflow_shifts_right, "5\n");
test_stdout!(with_negative_with_underflow_returns_zero, "0\n");
test_stdout!(
    with_positive_without_overflow_returns_small_integer,
    "true\n5\n"
);
test_stdout!(
    with_positive_with_overflow_returns_big_integer,
    "true\n18446744073709551616\n"
);
