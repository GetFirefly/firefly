use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_small_integer_or_big_integer_or_float_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::float(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be a number", |v| !v.is_number()),
                ),
                |(left, right)| {
                    prop_assert_eq!(native(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_float_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::float(arc_process.clone()), |operand| {
                prop_assert_eq!(native(operand, operand), false.into());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_same_value_float_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &any::<f64>().prop_map(|f| {
                    let mut heap = arc_process.acquire_heap();

                    (heap.float(f).unwrap(), heap.float(f).unwrap())
                }),
                |(left, right)| {
                    prop_assert_eq!(native(left.into(), right.into()), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_float_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::float(arc_process.clone()),
                    strategy::term::float(arc_process.clone()),
                )
                    .prop_filter("Right and left must be different", |(left, right)| {
                        left != right
                    }),
                |(left, right)| {
                    prop_assert_eq!(native(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_value_small_integer_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::small_integer_float_integral_i64().prop_map(|i| {
                    let mut heap = arc_process.acquire_heap();

                    (heap.float(i as f64).unwrap(), heap.integer(i).unwrap())
                }),
                |(left, right)| {
                    prop_assert_eq!(native(left.into(), right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_value_small_integer_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::small_integer_float_integral_i64().prop_map(|i| {
                    let mut heap = arc_process.acquire_heap();

                    (heap.float(i as f64).unwrap(), heap.integer(i + 1).unwrap())
                }),
                |(left, right)| {
                    prop_assert_eq!(native(left.into(), right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_value_big_integer_right_returns_false() {
    match strategy::term::big_integer_float_integral_i64() {
        Some(strategy) => {
            with_process_arc(|arc_process| {
                TestRunner::new(Config::with_source_file(file!()))
                    .run(
                        &strategy.prop_map(|i| {
                            let mut heap = arc_process.acquire_heap();

                            (heap.float(i as f64).unwrap(), heap.integer(i).unwrap())
                        }),
                        |(left, right)| {
                            prop_assert_eq!(native(left.into(), right), false.into());

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
fn with_different_value_big_integer_right_returns_true() {
    match strategy::term::big_integer_float_integral_i64() {
        Some(strategy) => {
            with_process_arc(|arc_process| {
                TestRunner::new(Config::with_source_file(file!()))
                    .run(
                        &strategy.prop_map(|i| {
                            let mut heap = arc_process.acquire_heap();

                            (heap.float(i as f64).unwrap(), heap.integer(i + 1).unwrap())
                        }),
                        |(left, right)| {
                            prop_assert_eq!(native(left.into(), right), true.into());

                            Ok(())
                        },
                    )
                    .unwrap();
            });
        }
        None => (),
    };
}
