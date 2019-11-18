use super::*;

use proptest::strategy::Strategy;

mod with_bit_count;
mod without_bit_count;

#[test]
fn without_integer_start_without_integer_length_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_bitstring(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(binary, start, length)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, start, length),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn without_integer_start_with_integer_length_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_bitstring(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(binary, start, length)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, start, length),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_start_without_integer_length_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_bitstring(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(binary, start, length)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, start, length),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_negative_start_with_valid_length_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_bitstring(arc_process.clone()),
                    strategy::term::integer::small::negative(arc_process.clone()),
                )
                    .prop_flat_map(|(binary, start)| {
                        (
                            Just(binary),
                            Just(start),
                            (0..=total_byte_len(binary))
                                .prop_map(|length| arc_process.integer(length).unwrap()),
                        )
                    }),
                |(binary, start, length)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, start, length),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_start_greater_than_size_with_non_negative_length_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_bitstring(arc_process.clone()).prop_flat_map(|binary| {
                    (
                        Just(binary),
                        Just(arc_process.integer(total_byte_len(binary) + 1).unwrap()),
                        (0..=total_byte_len(binary))
                            .prop_map(|length| arc_process.integer(length).unwrap()),
                    )
                }),
                |(binary, start, length)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, start, length),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_start_less_than_size_with_negative_length_past_start_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_bitstring::with_byte_len_range(
                    strategy::NON_EMPTY_RANGE_INCLUSIVE.into(),
                    arc_process.clone(),
                )
                .prop_flat_map(|binary| (Just(binary), 0..total_byte_len(binary)))
                .prop_map(|(binary, start)| {
                    let mut heap = arc_process.acquire_heap();

                    (
                        binary,
                        heap.integer(start).unwrap(),
                        heap.integer(-((start as isize) + 1)).unwrap(),
                    )
                }),
                |(binary, start, length)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, start, length),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_start_less_than_size_with_positive_length_past_end_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_bitstring::with_byte_len_range(
                    strategy::NON_EMPTY_RANGE_INCLUSIVE.into(),
                    arc_process.clone(),
                )
                .prop_flat_map(|binary| (Just(binary), 0..total_byte_len(binary)))
                .prop_map(|(binary, start)| {
                    let mut heap = arc_process.acquire_heap();

                    (
                        binary,
                        heap.integer(start).unwrap(),
                        heap.integer(total_byte_len(binary) - start + 1).unwrap(),
                    )
                }),
                |(binary, start, length)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, start, length),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_positive_start_and_negative_length_returns_subbinary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_bitstring::with_byte_len_range(
                    (2..=4).into(),
                    arc_process.clone(),
                )
                .prop_flat_map(|binary| {
                    let byte_len = total_byte_len(binary);

                    (Just(binary), (1..byte_len))
                })
                .prop_flat_map(|(binary, start)| {
                    (Just(binary), Just(start), (-(start as isize))..=(-1))
                })
                .prop_map(|(binary, start, length)| {
                    let mut heap = arc_process.acquire_heap();

                    (
                        binary,
                        heap.integer(start).unwrap(),
                        heap.integer(length).unwrap(),
                    )
                }),
                |(binary, start, length)| {
                    let result = native(&arc_process, binary, start, length);

                    prop_assert!(result.is_ok());

                    let returned = result.unwrap();

                    prop_assert!(returned.is_boxed_subbinary());

                    Ok(())
                },
            )
            .unwrap();
    });
}
