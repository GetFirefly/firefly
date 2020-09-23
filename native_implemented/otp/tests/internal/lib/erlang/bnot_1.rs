test_stdout!(without_integer_errors_badarith, "{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n");
test_stdout!(with_small_integer_returns_small_integer, "true\ntrue\n-3\n");
test_stdout!(
    with_big_integer_returns_big_integer,
    "true\ntrue\n-12297829382473034411\n"
);
