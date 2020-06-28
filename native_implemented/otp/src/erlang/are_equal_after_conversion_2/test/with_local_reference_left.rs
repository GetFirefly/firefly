use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_local_reference_right_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::local_reference(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Right cannot be a local reference", |right| {
                        !right.is_boxed_local_reference()
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
fn with_same_local_reference_right_returns_true() {
    run!(
        |arc_process| strategy::term::local_reference(arc_process.clone()),
        |operand| {
            prop_assert_eq!(result(operand, operand), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_local_reference_right_returns_false() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), proptest::prelude::any::<u64>()).prop_map(
                |(arc_process, number)| {
                    (
                        arc_process.reference(number).unwrap(),
                        arc_process.reference(number + 1).unwrap(),
                    )
                },
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}
