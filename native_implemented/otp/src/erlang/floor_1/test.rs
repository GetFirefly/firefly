use std::convert::TryInto;

use num_bigint::BigInt;

use num_traits::Num;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::{Encoded, Float};

use crate::erlang::floor_1::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_number_errors_badarg() {
    crate::test::without_number_errors_badarg(file!(), result);
}

#[test]
fn with_integer_returns_integer() {
    crate::test::with_integer_returns_integer(file!(), result);
}

#[test]
fn with_float_rounds_down_to_previous_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::float(arc_process.clone()), |number| {
                let result = result(&arc_process, number);

                prop_assert!(result.is_ok());

                let result_term = result.unwrap();

                prop_assert!(result_term.is_integer());

                let number_float: Float = number.try_into().unwrap();
                let number_f64: f64 = number_float.into();

                if number_f64.fract() == 0.0 {
                    // f64::to_string() has no decimal point when there is no `fract`.
                    let number_big_int =
                        <BigInt as Num>::from_str_radix(&number_f64.to_string(), 10).unwrap();
                    let result_big_int: BigInt = result_term.try_into().unwrap();

                    prop_assert_eq!(number_big_int, result_big_int);
                } else {
                    prop_assert!(result_term <= number, "{:?} <= {:?}", result_term, number);
                }

                Ok(())
            })
            .unwrap();
    });
}
