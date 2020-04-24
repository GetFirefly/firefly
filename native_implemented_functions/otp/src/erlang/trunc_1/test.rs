use std::convert::TryInto;

use proptest::prop_assert_eq;

use crate::erlang::trunc_1::result;

#[test]
fn without_number_errors_badarg() {
    crate::test::without_number_errors_badarg(file!(), result);
}

#[test]
fn with_integer_returns_integer() {
    crate::test::with_integer_returns_integer(file!(), result);
}

#[test]
fn with_float_truncates_to_integer() {
    crate::test::number_to_integer_with_float(file!(), result, |_, number_f64, result_term| {
        let result_f64: f64 = result_term.try_into().unwrap();
        let number_fract = number_f64.fract();

        prop_assert_eq!(result_f64, number_f64 - number_fract);

        Ok(())
    });
}
