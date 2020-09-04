test_stdout!(without_number_addend_errors_badarith, "{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n");

test_stdout!(
    with_zero_small_integer_returns_same_big_integer,
    "true\ntrue\ntrue\ntrue\n"
);
test_stdout!(
    that_is_positive_with_positive_small_integer_addend_returns_greater_big_integer,
    "true\ntrue\ntrue\ntrue\ntrue\ntrue\n"
);
test_stdout!(
    that_is_positive_with_positive_big_integer_addend_returns_greater_big_integer,
    "true\ntrue\ntrue\ntrue\ntrue\ntrue\n"
);
test_stdout!(
    with_float_addend_without_underflow_or_overflow_returns_float,
    "true\ntrue\n"
);
test_stdout!(with_float_addend_with_underflow_returns_min_float, "true\ntrue\n-179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0\n");
test_stdout!(with_float_addend_with_overflow_returns_max_float, "true\ntrue\n179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0\n");
