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
                        erlang::binary_part_3(binary, start, length, &arc_process),
                        Err(badarg!())
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
                        erlang::binary_part_3(binary, start, length, &arc_process),
                        Err(badarg!())
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
                        erlang::binary_part_3(binary, start, length, &arc_process),
                        Err(badarg!())
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
                            (0..=binary.byte_len())
                                .prop_map(|length| length.into_process(&arc_process)),
                        )
                    }),
                |(binary, start, length)| {
                    prop_assert_eq!(
                        erlang::binary_part_3(binary, start, length, &arc_process),
                        Err(badarg!())
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
                        Just((binary.byte_len() + 1).into_process(&arc_process)),
                        (0..=binary.byte_len())
                            .prop_map(|length| length.into_process(&arc_process)),
                    )
                }),
                |(binary, start, length)| {
                    prop_assert_eq!(
                        erlang::binary_part_3(binary, start, length, &arc_process),
                        Err(badarg!())
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
                .prop_flat_map(|binary| (Just(binary), 0..binary.byte_len()))
                .prop_map(|(binary, start)| {
                    (
                        binary,
                        start.into_process(&arc_process),
                        (-((start as isize) + 1)).into_process(&arc_process),
                    )
                }),
                |(binary, start, length)| {
                    prop_assert_eq!(
                        erlang::binary_part_3(binary, start, length, &arc_process),
                        Err(badarg!())
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
                .prop_flat_map(|binary| (Just(binary), 0..binary.byte_len()))
                .prop_map(|(binary, start)| {
                    (
                        binary,
                        start.into_process(&arc_process),
                        (binary.byte_len() - start + 1).into_process(&arc_process),
                    )
                }),
                |(binary, start, length)| {
                    prop_assert_eq!(
                        erlang::binary_part_3(binary, start, length, &arc_process),
                        Err(badarg!())
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
                    let byte_len = binary.byte_len();

                    (Just(binary), (1..byte_len))
                })
                .prop_flat_map(|(binary, start)| {
                    (Just(binary), Just(start), (-(start as isize))..=(-1))
                })
                .prop_map(|(binary, start, length)| {
                    (
                        binary,
                        start.into_process(&arc_process),
                        length.into_process(&arc_process),
                    )
                }),
                |(binary, start, length)| {
                    let result = erlang::binary_part_3(binary, start, length, &arc_process);

                    prop_assert!(result.is_ok());

                    let returned_boxed = result.unwrap();

                    prop_assert_eq!(returned_boxed.tag(), Boxed);

                    let returned_unboxed: &Term = returned_boxed.unbox_reference();

                    prop_assert_eq!(returned_unboxed.tag(), Subbinary);

                    Ok(())
                },
            )
            .unwrap();
    });
}
