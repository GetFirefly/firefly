test_stdout!(with_small_integer_returns_small_integer, "-1\n0\n1\n");
test_stdout!(
    with_big_integer_returns_big_integer,
    "-9223372036854775808\n9223372036854775807\n"
);
