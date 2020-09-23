#[path = "bor_2/with_big_integer_left.rs"]
mod with_big_integer_left;
#[path = "bor_2/with_small_integer_left.rs"]
mod with_small_integer_left;

test_stdout!(without_integer_left_errors_badarith, "{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n");
test_stdout!(with_integer_left_without_integer_right_errors_badarith, "{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n");
test_stdout!(with_same_integer_returns_same_integer, "true\ntrue\n");
