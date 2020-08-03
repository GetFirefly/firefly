use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};

use crate::erlang::float_1::result;
use crate::test::strategy;

#[test]
fn without_number_errors_badarg() {
    crate::test::without_number_errors_badarg(file!(), result);
}

#[test]
fn with_integer_returns_float_with_same_value() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                -9007199254740992_i64..=9007199254740992_i64,
            )
                .prop_map(|(arc_process, i)| {
                    (
                        arc_process.clone(),
                        arc_process.integer(i).unwrap(),
                        arc_process.float(i as f64).unwrap(),
                    )
                })
        },
        |(arc_process, number, float)| {
            prop_assert_eq!(result(&arc_process, number), Ok(float));

            Ok(())
        },
    );
}

#[test]
fn with_float_returns_same_float() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::float(arc_process.clone()),
            )
        },
        |(arc_process, number)| {
            prop_assert_eq!(result(&arc_process, number), Ok(number));

            Ok(())
        },
    );
}
