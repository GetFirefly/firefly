use std::convert::TryInto;

use num_bigint::BigInt;

use num_traits::Num;

use proptest::strategy::Just;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::{Encoded, Float};

use crate::otp::erlang::round_1::native;
use crate::test::strategy;

#[test]
fn without_number_errors_badarg() {
    crate::test::without_number_errors_badarg(file!(), native);
}

#[test]
fn with_integer_returns_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, number)| {
            prop_assert_eq!(native(&arc_process, number), Ok(number));

            Ok(())
        },
    );
}

#[test]
fn with_float_rounds_to_nearest_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::float(arc_process.clone()),
            )
        },
        |(arc_process, number)| {
            let result = native(&arc_process, number);

            prop_assert!(result.is_ok());

            let result_term = result.unwrap();

            prop_assert!(result_term.is_integer());

            let number_float: Float = number.try_into().unwrap();
            let number_f64: f64 = number_float.into();
            let number_fract = number_f64.fract();

            if number_fract == 0.0 {
                // f64::to_string() has no decimal point when there is no `fract`.
                let number_big_int =
                    <BigInt as Num>::from_str_radix(&number_f64.to_string(), 10).unwrap();
                let result_big_int: BigInt = result_term.try_into().unwrap();

                prop_assert_eq!(number_big_int, result_big_int);
            } else {
                let result_f64: f64 = result_term.try_into().unwrap();

                prop_assert!((result_f64 - number_f64).abs() <= 0.5)
            }

            Ok(())
        },
    );
}
