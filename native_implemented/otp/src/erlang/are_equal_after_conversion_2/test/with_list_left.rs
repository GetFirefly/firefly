use super::*;

use proptest::collection::SizeRange;
use proptest::strategy::Strategy;

use crate::test::strategy::NON_EMPTY_RANGE_INCLUSIVE;

#[test]
fn without_list_right_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::list::non_empty_maybe_improper(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Right must not be list", |v| !v.is_list()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_list_right_returns_true() {
    run!(
        |arc_process| strategy::term::list::non_empty_maybe_improper(arc_process.clone()),
        |operand| {
            prop_assert_eq!(result(operand, operand), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_value_list_right_returns_true() {
    run!(
        |arc_process| {
            let size_range: SizeRange = NON_EMPTY_RANGE_INCLUSIVE.clone().into();

            proptest::collection::vec(strategy::term(arc_process.clone()), size_range).prop_map(
                move |vec| {
                    let mut heap = arc_process.acquire_heap();

                    match vec.len() {
                        1 => (
                            heap.list_from_slice(&vec)
                                .unwrap()
                                .map(|l| l.into())
                                .unwrap_or(Term::NIL),
                            heap.list_from_slice(&vec)
                                .unwrap()
                                .map(|l| l.into())
                                .unwrap_or(Term::NIL),
                        ),
                        len => {
                            let last_index = len - 1;

                            (
                                heap.improper_list_from_slice(&vec[0..last_index], vec[last_index])
                                    .unwrap()
                                    .map(|l| l.into())
                                    .unwrap_or(Term::NIL),
                                heap.improper_list_from_slice(&vec[0..last_index], vec[last_index])
                                    .unwrap()
                                    .map(|l| l.into())
                                    .unwrap_or(Term::NIL),
                            )
                        }
                    }
                },
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left.into(), right.into()), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_list_right_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::list::non_empty_maybe_improper(arc_process.clone()),
                strategy::term::list::non_empty_maybe_improper(arc_process.clone()),
            )
                .prop_filter("Lists must be different", |(left, right)| left != right)
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}
