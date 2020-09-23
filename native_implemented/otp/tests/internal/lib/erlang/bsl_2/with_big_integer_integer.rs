test_stdout!(
    with_negative_without_big_integer_underflow_shifts_right_and_returns_big_integer,
    "true\n1656137896166982451072\n"
);
test_stdout!(with_negative_with_big_integer_underflow_without_small_integer_underflow_shifts_right_and_returns_small_integer, "true\n11\n");
test_stdout!(with_negative_with_underflow_returns_zero, "0\n");
test_stdout!(
    with_positive_returns_big_integer,
    "true\n6624551584667929804288\n"
);
