test_stdout!(
    with_small_integer_divisor_returns_small_integer,
    "true\n1024\n"
);
test_stdout!(with_big_integer_divisor_returns_zero, "false\n0\n");
