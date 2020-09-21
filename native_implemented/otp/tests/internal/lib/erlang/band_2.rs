#[path = "band_2/with_big_integer_left.rs"]
mod with_big_integer_left;
#[path = "band_2/with_small_integer_left.rs"]
mod with_small_integer_left;

test_stdout!(without_integer_right_errors_badarith, "{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n");
