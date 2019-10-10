use std::convert::TryInto;

use num_bigint::BigInt;

use num_traits::Num;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::Float;

use crate::otp::erlang::round_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_number_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_number(arc_process.clone()),
                |number| {
                    prop_assert_eq!(native(&arc_process, number), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_returns_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_integer(arc_process.clone()), |number| {
                prop_assert_eq!(native(&arc_process, number), Ok(number));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_float_rounds_to_nearest_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::float(arc_process.clone()), |number| {
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
                } else if (0.0 < number_f64 && number_fract < 0.5)
                    || (number_f64 < 0.0 && number_fract < -0.5)
                {
                    prop_assert!(
                        result_term <= number,
                        "Expected rounded ({:?}) <= number ({:?})",
                        result_term,
                        number
                    );
                } else {
                    prop_assert!(
                        number <= result_term,
                        "Expected number ({:?}) <= rounded ({:?})",
                        number,
                        result_term
                    );
                }

                Ok(())
            })
            .unwrap();
    });
}
