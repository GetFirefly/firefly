test_stdout!(
    with_big_integer_right_returns_big_integer,
    "12297829382473034414\ntrue\n"
);
test_stdout!(with_small_integer_right_returns_big_integer, "14\ntrue\n");
