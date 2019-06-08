use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_tuple_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_tuple(arc_process.clone()),
                    strategy::term::atom(),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(tuple, record_tag, size)| {
                    prop_assert_eq!(
                        erlang::is_record_3(tuple, record_tag, size),
                        Ok(false.into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
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
                    prop_assert_eq!(erlang::is_record_3(tuple, record_tag, size), Err(badarg!()));

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
                            !(size.is_integer() && &0.into_process(&arc_process) <= size)
                        }),
                ),
                |(record_tag, size)| {
                    let tuple = Term::slice_to_tuple(&[], &arc_process);

                    prop_assert_eq!(erlang::is_record_3(tuple, record_tag, size), Err(badarg!()));

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
                    let tuple = Term::slice_to_tuple(&[], &arc_process);

                    prop_assert_eq!(
                        erlang::is_record_3(tuple, record_tag, size),
                        Ok(false.into())
                    );

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
                            let size = tail_element_vec.len().into_process(&arc_process);

                            (
                                Term::slice_to_tuple(&tail_element_vec, &arc_process),
                                tested_record_tag,
                                size,
                            )
                        },
                    ),
                |(tuple, record_tag, size)| {
                    prop_assert_eq!(
                        erlang::is_record_3(tuple, record_tag, size),
                        Ok(false.into())
                    );

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
                        let tuple_size = tail_element_vec.len().into_process(&arc_process);

                        (
                            Just(Term::slice_to_tuple(&tail_element_vec, &arc_process)),
                            Just(record_tag),
                            strategy::term::integer::non_negative(arc_process.clone())
                                .prop_filter("Size cannot match tuple size", move |size| {
                                    size != &tuple_size
                                }),
                        )
                    }),
                |(tuple, record_tag, size)| {
                    prop_assert_eq!(
                        erlang::is_record_3(tuple, record_tag, size),
                        Ok(false.into())
                    );

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
                        let size = tail_element_vec.len().into_process(&arc_process);

                        (
                            Term::slice_to_tuple(&tail_element_vec, &arc_process),
                            record_tag,
                            size,
                        )
                    }),
                |(tuple, record_tag, size)| {
                    prop_assert_eq!(
                        erlang::is_record_3(tuple, record_tag, size),
                        Ok(true.into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
