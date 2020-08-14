use crate::erlang::float_1::result;

#[test]
fn without_number_errors_badarg() {
    crate::test::without_number_errors_badarg(file!(), result);
}

// `with_integer_returns_float_with_same_value` in integration tests
// `with_float_returns_same_float` in integration tests
