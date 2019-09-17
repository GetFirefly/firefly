use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::alloc::heap_alloc::HeapAlloc;

use crate::otp::erlang::is_record_3::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_tuple_returns_false() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    strategy::term::is_not_tuple(arc_process.clone()),
                    strategy::term::atom(),
                    strategy::term::is_integer(arc_process),
                )
            }),
            |(tuple, record_tag, size)| {
                prop_assert_eq!(native(tuple, record_tag, size), Ok(false.into()));

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_tuple_without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(tuple, record_tag, size)| {
                    prop_assert_eq!(native(tuple, record_tag, size), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_tuple_with_atom_without_non_negative_size_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term(arc_process.clone())
                        .prop_filter("Size must not be a non-negative integer", |size| {
                            !(size.is_integer() && &arc_process.integer(0).unwrap() <= size)
                        }),
                ),
                |(record_tag, size)| {
                    let tuple = arc_process.tuple_from_slice(&[]).unwrap();

                    prop_assert_eq!(native(tuple, record_tag, size), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_tuple_with_atom_with_non_negative_size_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term::integer::non_negative(arc_process.clone()),
                ),
                |(record_tag, size)| {
                    let tuple = arc_process.tuple_from_slice(&[]).unwrap();

                    prop_assert_eq!(native(tuple, record_tag, size), Ok(false.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_non_empty_tuple_without_record_tag_with_size_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term::atom(), strategy::term::atom())
                    .prop_filter(
                        "Actual and tested record tag must be different",
                        |(actual_record_tag, tested_record_tag)| {
                            actual_record_tag != tested_record_tag
                        },
                    )
                    .prop_flat_map(|(actual_record_tag, tested_record_tag)| {
                        (
                            Just(actual_record_tag),
                            proptest::collection::vec(
                                strategy::term(arc_process.clone()),
                                strategy::size_range(),
                            ),
                            Just(tested_record_tag),
                        )
                    })
                    .prop_map(
                        |(actual_record_tag, mut tail_element_vec, tested_record_tag)| {
                            tail_element_vec.insert(0, actual_record_tag);

                            let mut heap = arc_process.acquire_heap();

                            let size = heap.integer(tail_element_vec.len()).unwrap();

                            (
                                heap.tuple_from_slice(&tail_element_vec).unwrap(),
                                tested_record_tag,
                                size,
                            )
                        },
                    ),
                |(tuple, record_tag, size)| {
                    prop_assert_eq!(native(tuple, record_tag, size), Ok(false.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_non_empty_tuple_with_record_tag_without_size_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    proptest::collection::vec(
                        strategy::term(arc_process.clone()),
                        strategy::size_range(),
                    ),
                )
                    .prop_flat_map(|(record_tag, mut tail_element_vec)| {
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
                    }),
                |(tuple, record_tag, size)| {
                    prop_assert_eq!(native(tuple, record_tag, size), Ok(false.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_non_empty_tuple_with_record_tag_with_size_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    proptest::collection::vec(
                        strategy::term(arc_process.clone()),
                        strategy::size_range(),
                    ),
                )
                    .prop_map(|(record_tag, mut tail_element_vec)| {
                        tail_element_vec.insert(0, record_tag);
                        let size = arc_process.integer(tail_element_vec.len()).unwrap();

                        (
                            arc_process.tuple_from_slice(&tail_element_vec).unwrap(),
                            record_tag,
                            size,
                        )
                    }),
                |(tuple, record_tag, size)| {
                    prop_assert_eq!(native(tuple, record_tag, size), Ok(true.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}
