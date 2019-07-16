use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_small_integer_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::small(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be a small integer", |v| !v.is_smallint()),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_not_equal_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_small_integer_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::small(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(
                        erlang::are_exactly_not_equal_2(operand, operand),
                        false.into()
                    );

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
                &(SmallInteger::MIN_VALUE..SmallInteger::MAX_VALUE).prop_map(move |i| {
                    let mut heap = arc_process.acquire_heap();

                    (heap.integer(i).unwrap(), heap.integer(i).unwrap())
                }),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_not_equal_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_small_integer_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(SmallInteger::MIN_VALUE..SmallInteger::MAX_VALUE).prop_map(move |i| {
                    let mut heap = arc_process.acquire_heap();

                    (heap.integer(i).unwrap(), heap.integer(i + 1).unwrap())
                }),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_not_equal_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}
