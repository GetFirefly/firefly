#[path = "add_2/with_big_integer_augend.rs"]
pub mod with_big_integer_augend;
#[path = "add_2/with_float_augend.rs"]
pub mod with_float_augend;
#[path = "add_2/with_small_integer_augend.rs"]
pub mod with_small_integer_augend;

test_stdout!(without_number_augend_errors_badarith, "{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n{caught, error, badarith}\n");
