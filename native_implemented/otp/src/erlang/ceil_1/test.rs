use proptest::prop_assert;

use crate::erlang::ceil_1::result;

#[test]
fn without_number_errors_badarg() {
    crate::test::without_number_errors_badarg(file!(), result);
}

#[test]
fn with_integer_returns_integer() {
    crate::test::with_integer_returns_integer(file!(), result);
}

#[test]
fn with_float_round_up_to_next_integer() {
    crate::test::number_to_integer_with_float(file!(), result, |number, _, result_term| {
        prop_assert!(number <= result_term, "{:?} <= {:?}", number, result_term);

        Ok(())
    })
}
