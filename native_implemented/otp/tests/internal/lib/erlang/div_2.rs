#[path = "div_2/with_big_integer_dividend.rs"]
mod with_big_integer_dividend;
#[path = "div_2/with_small_integer_dividend.rs"]
mod with_small_integer_dividend;

test_stdout!(without_integer_dividend_errors_badarith, "{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n");
test_stdout!(
    with_integer_dividend_without_integer_divisor_errors_badarith,
    "{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n"
);
test_stdout!(
    with_integer_dividend_with_zero_divisor_errors_badarith,
    "{caught, error, badarith}\n{caught, error, badarith}\n"
);
