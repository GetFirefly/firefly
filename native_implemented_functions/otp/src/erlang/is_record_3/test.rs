use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::is_record_3::result;
use crate::test::strategy;

#[test]
fn without_tuple_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::is_not_tuple(arc_process.clone()),
                strategy::term::atom(),
                strategy::term::is_integer(arc_process),
            )
        },
        |(tuple, record_tag, size)| {
            prop_assert_eq!(result(tuple, record_tag, size), Ok(false.into()));

            Ok(())
        },
    );
}

#[test]
fn with_tuple_without_atom_errors_badarg() {
    run!(
        |arc_process| {
            (
                strategy::term::tuple(arc_process.clone()),
                strategy::term::is_not_atom(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(tuple, record_tag, size)| {
            prop_assert_is_not_atom!(result(tuple, record_tag, size), "record tag", record_tag);

            Ok(())
        },
    );
}

#[test]
fn with_empty_tuple_with_atom_without_non_negative_size_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::atom(),
                (
                    Just(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                )
                    .prop_filter_map(
                        "Size must not be a non-negative integer",
                        |(arc_process, size)| {
                            if !(size.is_integer() && arc_process.integer(0).unwrap() <= size) {
                                Some(size)
                            } else {
                                None
                            }
                        },
                    ),
            )
        },
        |(arc_process, record_tag, size)| {
            let tuple = arc_process.tuple_from_slice(&[]).unwrap();

            prop_assert_badarg!(
                result(tuple, record_tag, size),
                format!("size ({}) must be a positive integer", size)
            );

            Ok(())
        },
    );
}

#[test]
fn with_empty_tuple_with_atom_with_non_negative_size_returns_false() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::atom(),
                strategy::term::integer::non_negative(arc_process.clone()),
            )
        },
        |(arc_process, record_tag, size)| {
            let tuple = arc_process.tuple_from_slice(&[]).unwrap();

            prop_assert_eq!(result(tuple, record_tag, size), Ok(false.into()));

            Ok(())
        },
    );
}

#[test]
fn with_non_empty_tuple_without_record_tag_with_size_returns_false() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::atom(),
                strategy::term::atom(),
            )
                .prop_filter(
                    "Actual and tested record tag must be different",
                    |(_, actual_record_tag, tested_record_tag)| {
                        actual_record_tag != tested_record_tag
                    },
                )
                .prop_flat_map(|(arc_process, actual_record_tag, tested_record_tag)| {
                    (
                        Just(arc_process.clone()),
                        Just(actual_record_tag),
                        proptest::collection::vec(
                            strategy::term(arc_process.clone()),
                            strategy::size_range(),
                        ),
                        Just(tested_record_tag),
                    )
                })
                .prop_map(
                    |(arc_process, actual_record_tag, mut tail_element_vec, tested_record_tag)| {
                        tail_element_vec.insert(0, actual_record_tag);

                        let size = arc_process.integer(tail_element_vec.len()).unwrap();

                        (
                            arc_process.tuple_from_slice(&tail_element_vec).unwrap(),
                            tested_record_tag,
                            size,
                        )
                    },
                )
        },
        |(tuple, record_tag, size)| {
            prop_assert_eq!(result(tuple, record_tag, size), Ok(false.into()));

            Ok(())
        },
    );
}

#[test]
fn with_non_empty_tuple_with_record_tag_without_size_returns_false() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::atom(),
                proptest::collection::vec(
                    strategy::term(arc_process.clone()),
                    strategy::size_range(),
                ),
            )
                .prop_flat_map(|(arc_process, record_tag, mut tail_element_vec)| {
                    tail_element_vec.insert(0, record_tag);
                    let tuple_size = arc_process.integer(tail_element_vec.len()).unwrap();

                    (
                        Just(arc_process.tuple_from_slice(&tail_element_vec).unwrap()),
                        Just(record_tag),
                        strategy::term::integer::non_negative(arc_process.clone())
                            .prop_filter("Size cannot match tuple size", move |size| {
                                size != &tuple_size
                            }),
                    )
                })
        },
        |(tuple, record_tag, size)| {
            prop_assert_eq!(result(tuple, record_tag, size), Ok(false.into()));

            Ok(())
        },
    );
}

#[test]
fn with_non_empty_tuple_with_record_tag_with_size_returns_true() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::atom(),
                proptest::collection::vec(
                    strategy::term(arc_process.clone()),
                    strategy::size_range(),
                ),
            )
                .prop_map(|(arc_process, record_tag, mut tail_element_vec)| {
                    tail_element_vec.insert(0, record_tag);
                    let size = arc_process.integer(tail_element_vec.len()).unwrap();

                    (
                        arc_process.tuple_from_slice(&tail_element_vec).unwrap(),
                        record_tag,
                        size,
                    )
                })
        },
        |(tuple, record_tag, size)| {
            prop_assert_eq!(result(tuple, record_tag, size), Ok(true.into()));

            Ok(())
        },
    );
}
