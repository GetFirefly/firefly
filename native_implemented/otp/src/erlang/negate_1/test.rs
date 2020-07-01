use proptest::strategy::{Just, Strategy};
use proptest::{prop_assert_eq, prop_oneof};

use crate::erlang::negate_1::result;
use crate::test::strategy;

#[test]
fn without_number_errors_badarith() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_number(arc_process.clone()),
            )
        },
        |(arc_process, number)| {
            prop_assert_badarith!(
                result(&arc_process, number),
                format!("number ({}) is neither an integer nor a float", number)
            );

            Ok(())
        },
    );
}

#[test]
fn with_integer_returns_integer_of_opposite_sign() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                prop_oneof![std::isize::MIN..=-1, 1..=std::isize::MAX],
            )
                .prop_map(|(arc_process, i)| {
                    (arc_process.clone(), arc_process.integer(i).unwrap(), i)
                })
        },
        |(arc_process, number, i)| {
            let negated = arc_process.integer(-i).unwrap();

            prop_assert_eq!(result(&arc_process, number), Ok(negated));

            Ok(())
        },
    );
}

#[test]
fn with_float_returns_float_of_opposite_sign() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                prop_oneof![std::f64::MIN..=-1.0, 1.0..=std::f64::MAX],
            )
                .prop_map(|(arc_process, f)| {
                    (arc_process.clone(), arc_process.float(f).unwrap(), f)
                })
        },
        |(arc_process, number, f)| {
            let negated = arc_process.float(-f).unwrap();

            prop_assert_eq!(result(&arc_process, number), Ok(negated));

            Ok(())
        },
    );
}
