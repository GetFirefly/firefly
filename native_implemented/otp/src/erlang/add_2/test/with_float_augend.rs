use super::*;

#[test]
fn without_number_addend_errors_badarith() {
    super::without_number_addend_errors_badarith(file!(), strategy::term::float);
}

// `with_small_integer_addend_returns_float` in integration tests
// `with_big_integer_addend_returns_float` in integration tests
// `with_float_addend_without_underflow_or_overflow_returns_float` in integration tests
// `with_float_addend_with_underflow_returns_min_float` in integration tests
// `with_float_addend_with_overflow_returns_max_float` in integration tests
