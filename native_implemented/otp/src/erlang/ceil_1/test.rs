use crate::erlang::ceil_1::result;

#[test]
fn without_number_errors_badarg() {
    crate::test::without_number_errors_badarg(file!(), result);
}

// `with_integer_returns_integer` in integration tests
// `with_float_round_up_to_next_integer` in integration tests
