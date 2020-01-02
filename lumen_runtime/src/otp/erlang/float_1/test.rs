use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::float_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_number_errors_badarg() {
    crate::test::without_number_errors_badarg(file!(), native);
}

#[test]
fn with_integer_returns_float_with_same_value() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(-9007199254740992_i64..=9007199254740992_i64).prop_map(|i| {
                    (
                        arc_process.integer(i).unwrap(),
                        arc_process.float(i as f64).unwrap(),
                    )
                }),
                |(number, float)| {
                    prop_assert_eq!(native(&arc_process, number), Ok(float));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_float_returns_same_float() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::float(arc_process.clone()), |number| {
                prop_assert_eq!(native(&arc_process, number), Ok(number));

                Ok(())
            })
            .unwrap();
    });
}
