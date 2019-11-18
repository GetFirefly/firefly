use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_big_integer_or_float_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::big(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be a big integer or float", |v| {
                            !(v.is_boxed_bigint() || v.is_boxed_float())
                        }),
                ),
                |(left, right)| {
                    prop_assert_eq!(native(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_big_integer_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::big(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(native(operand, operand), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_big_integer_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::big(arc_process.clone()),
                    strategy::term::integer::big(arc_process.clone()),
                )
                    .prop_filter("Right and left must be different", |(left, right)| {
                        left != right
                    }),
                |(left, right)| {
                    prop_assert_eq!(native(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_value_float_right_returns_true() {
    match strategy::term::big_integer_float_integral_i64() {
        Some(strategy) => {
            with_process_arc(|arc_process| {
                TestRunner::new(Config::with_source_file(file!()))
                    .run(
                        &strategy.prop_map(|i| {
                            let mut heap = arc_process.acquire_heap();

                            (heap.integer(i).unwrap(), heap.float(i as f64).unwrap())
                        }),
                        |(left, right)| {
                            prop_assert_eq!(native(left, right.into()), true.into());

                            Ok(())
                        },
                    )
                    .unwrap();
            });
        }
        None => (),
    };
}

#[test]
fn with_different_value_float_right_returns_false() {
    match strategy::term::big_integer_float_integral_i64() {
        Some(strategy) => {
            with_process_arc(|arc_process| {
                TestRunner::new(Config::with_source_file(file!()))
                    .run(
                        &strategy.prop_map(|i| {
                            let mut heap = arc_process.acquire_heap();

                            (heap.integer(i + 1).unwrap(), heap.float(i as f64).unwrap())
                        }),
                        |(left, right)| {
                            prop_assert_eq!(native(left, right.into()), false.into());

                            Ok(())
                        },
                    )
                    .unwrap();
            });
        }
        None => (),
    };
}
