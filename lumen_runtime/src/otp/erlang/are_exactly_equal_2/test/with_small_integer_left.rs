use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_small_integer_returns_false() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    strategy::term::integer::small(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be a small integer", |v| !v.is_smallint()),
                )
            }),
            |(left, right)| {
                prop_assert_eq!(native(left, right), false.into());

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_same_small_integer_right_returns_true() {
    run(
        file!(),
        |arc_process| strategy::term::integer::small(arc_process.clone()),
        |operand| {
            prop_assert_eq!(native(operand, operand), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_value_small_integer_right_returns_true() {
    run(
        file!(),
        |arc_process| {
            (SmallInteger::MIN_VALUE..SmallInteger::MAX_VALUE).prop_map(move |i| {
                (
                    arc_process.integer(i).unwrap(),
                    arc_process.integer(i).unwrap(),
                )
            })
        },
        |(left, right)| {
            prop_assert_eq!(native(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_small_integer_right_returns_false() {
    run(
        file!(),
        |arc_process| {
            (SmallInteger::MIN_VALUE..SmallInteger::MAX_VALUE).prop_map(move |i| {
                (
                    arc_process.integer(i).unwrap(),
                    arc_process.integer(i + 1).unwrap(),
                )
            })
        },
        |(left, right)| {
            prop_assert_eq!(native(left, right), false.into());

            Ok(())
        },
    );
}
