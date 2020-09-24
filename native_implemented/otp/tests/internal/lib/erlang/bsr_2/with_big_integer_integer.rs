test_stdout!(
    with_negative_shifts_left_and_returns_big_integer,
    "true\n6624551584667929804288\n"
);
test_stdout!(
    with_positive_with_big_integer_underflow_without_small_integer_underflow_returns_small_integer,
    "true\n1\n"
);
test_stdout!(with_positive_with_underflow_returns_zero, "0\n");
