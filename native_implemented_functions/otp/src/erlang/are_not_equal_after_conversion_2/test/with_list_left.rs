use super::*;

use proptest::collection::SizeRange;
use proptest::strategy::Strategy;

use crate::test::strategy::NON_EMPTY_RANGE_INCLUSIVE;

#[test]
fn without_list_right_returns_true() {
    run!(
        |arc_process| {
            (
                strategy::term::list::non_empty_maybe_improper(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Right must not be list", |v| !v.is_non_empty_list()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_list_right_returns_false() {
    run!(
        |arc_process| strategy::term::list::non_empty_maybe_improper(arc_process.clone()),
        |operand| {
            prop_assert_eq!(result(operand, operand), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_value_list_right_returns_false() {
    run!(
        |arc_process| {
            let size_range: SizeRange = NON_EMPTY_RANGE_INCLUSIVE.clone().into();

            proptest::collection::vec(strategy::term(arc_process.clone()), size_range).prop_map(
                move |vec| match vec.len() {
                    1 => (
                        arc_process.list_from_slice(&vec).unwrap(),
                        arc_process.list_from_slice(&vec).unwrap(),
                    ),
                    len => {
                        let last_index = len - 1;

                        (
                            arc_process
                                .improper_list_from_slice(&vec[0..last_index], vec[last_index])
                                .unwrap(),
                            arc_process
                                .improper_list_from_slice(&vec[0..last_index], vec[last_index])
                                .unwrap(),
                        )
                    }
                },
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_list_right_returns_true() {
    run!(
        |arc_process| {
            (
                strategy::term::list::non_empty_maybe_improper(arc_process.clone()),
                strategy::term::list::non_empty_maybe_improper(arc_process.clone()),
            )
                .prop_filter("Lists must be different", |(left, right)| left != right)
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}
