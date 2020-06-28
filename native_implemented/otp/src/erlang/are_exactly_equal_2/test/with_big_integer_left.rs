use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_big_integer_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::integer::big(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Right must not be a big integer", |v| !v.is_boxed_bigint()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_big_integer_returns_true() {
    run!(
        |arc_process| strategy::term::integer::big(arc_process.clone()),
        |operand| {
            prop_assert_eq!(result(operand, operand), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_big_integer_right_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::integer::big(arc_process.clone()),
                strategy::term::integer::big(arc_process.clone()),
            )
                .prop_filter("Right and left must be different", |(left, right)| {
                    left != right
                })
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}
