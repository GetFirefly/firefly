use proptest::prop_assert;

use crate::otp::erlang::ceil_1::native;

#[test]
fn without_number_errors_badarg() {
    crate::test::without_number_errors_badarg(file!(), native);
}

#[test]
fn with_integer_returns_integer() {
    crate::test::with_integer_returns_integer(file!(), native);
}

#[test]
fn with_float_round_up_to_next_integer() {
    crate::test::number_to_integer_with_float(file!(), native, |number, _, result_term| {
        prop_assert!(number <= result_term, "{:?} <= {:?}", number, result_term);

        Ok(())
    })
}
