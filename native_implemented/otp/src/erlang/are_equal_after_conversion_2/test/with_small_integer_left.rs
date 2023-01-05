use super::*;

#[test]
fn without_small_integer_or_float_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::integer::small(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Right must not be a small integer or float", |v| {
                        !(v.is_smallint() || v.is_float())
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
fn with_same_small_integer_right_returns_true() {
    run!(
        |arc_process| strategy::term::integer::small(arc_process.clone()),
        |operand| {
            prop_assert_eq!(result(operand, operand), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_value_small_integer_right_returns_true() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                Integer::MIN_SMALL..Integer::MAX_SMALL,
            )
                .prop_map(|(arc_process, i)| {
                    let mut heap = arc_process.acquire_heap();

                    (heap.integer(i).unwrap(), heap.integer(i).unwrap())
                })
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_small_integer_right_returns_false() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                Integer::MIN_SMALL..Integer::MAX_SMALL,
            )
                .prop_map(|(arc_process, i)| {
                    let mut heap = arc_process.acquire_heap();

                    (heap.integer(i).unwrap(), heap.integer(i + 1).unwrap())
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
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::small_integer_float_integral_i64(),
            )
                .prop_map(|(arc_process, i)| {
                    let mut heap = arc_process.acquire_heap();

                    (heap.integer(i).unwrap(), heap.into().unwrap())
                })
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right.into()), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_value_float_right_returns_false() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::small_integer_float_integral_i64(),
            )
                .prop_map(|(arc_process, i)| {
                    let mut heap = arc_process.acquire_heap();
                    // change float toward zero to ensure it remains in integral range
                    let diff = if i < 0 { 1 } else { -1 };

                    (
                        heap.integer(i).unwrap(),
                        ((i + diff) as f64).into(),
                    )
                })
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right.into()), false.into());

            Ok(())
        },
    );
}
