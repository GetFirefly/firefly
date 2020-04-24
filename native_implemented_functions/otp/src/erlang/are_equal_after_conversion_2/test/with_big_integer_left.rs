use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_big_integer_or_float_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::integer::big(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Right must not be a big integer or float", |v| {
                        !(v.is_boxed_bigint() || v.is_boxed_float())
                    }),
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

#[test]
fn with_same_value_float_right_returns_true() {
    match strategy::term::big_integer_float_integral_i64() {
        Some(ref strategy) => {
            run!(
                |arc_process| {
                    (Just(arc_process.clone()), strategy).prop_map(|(arc_process, i)| {
                        (
                            arc_process.integer(i).unwrap(),
                            arc_process.float(i as f64).unwrap(),
                        )
                    })
                },
                |(left, right)| {
                    prop_assert_eq!(result(left, right.into()), true.into());

                    Ok(())
                },
            );
        }
        None => (),
    };
}

#[test]
fn with_different_value_float_right_returns_false() {
    match strategy::term::big_integer_float_integral_i64() {
        Some(ref strategy) => {
            run!(
                |arc_process| {
                    (Just(arc_process.clone()), strategy).prop_map(|(arc_process, i)| {
                        (
                            arc_process.integer(i + 1).unwrap(),
                            arc_process.float(i as f64).unwrap(),
                        )
                    })
                },
                |(left, right)| {
                    prop_assert_eq!(result(left, right.into()), false.into());

                    Ok(())
                },
            );
        }
        None => (),
    };
}
