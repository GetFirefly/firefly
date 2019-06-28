use super::*;
use crate::otp::erlang::tests::strategy::term::binary;
use crate::otp::erlang::tests::strategy::term::binary::sub::{bit_offset, byte_count, byte_offset};
use crate::otp::erlang::tests::strategy::NON_EMPTY_RANGE_INCLUSIVE;

#[test]
fn with_positive_start_and_positive_length_returns_subbinary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &binary::sub::with_size_range(
                    byte_offset(),
                    bit_offset(),
                    (3_usize..=6_usize).boxed(),
                    (1_u8..=7_u8).boxed(),
                    arc_process.clone(),
                )
                .prop_flat_map(|binary| {
                    let byte_count = binary.unbox_reference::<sub::Binary>().byte_count;

                    // `start` must be 2 less than `byte_count` so that `length` can be at least 1
                    // and still get a full byte
                    (Just(binary), (1..=(byte_count - 2)))
                })
                .prop_flat_map(|(binary, start)| {
                    (
                        Just(binary),
                        Just(start),
                        1..=(binary.unbox_reference::<sub::Binary>().byte_count - start),
                    )
                })
                .prop_map(|(binary, start, length)| {
                    (
                        binary,
                        start.into_process(&arc_process),
                        length.into_process(&arc_process),
                    )
                }),
                |(binary, start, length)| {
                    let start_length = Term::slice_to_tuple(&[start, length], &arc_process);

                    let result = erlang::binary_part_2(binary, start_length, &arc_process);

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

#[test]
fn with_byte_count_start_and_negative_byte_count_length_returns_subbinary_without_bit_count() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &binary::sub::with_size_range(
                    byte_offset(),
                    bit_offset(),
                    NON_EMPTY_RANGE_INCLUSIVE.boxed(),
                    (1_u8..=7u8).boxed(),
                    arc_process.clone(),
                )
                .prop_map(|binary| {
                    let byte_count = binary.unbox_reference::<sub::Binary>().byte_count;

                    (
                        binary,
                        byte_count.into_process(&arc_process),
                        (-(byte_count as isize)).into_process(&arc_process),
                    )
                }),
                |(binary, start, length)| {
                    let expected_returned_binary_bytes: Vec<u8> = binary
                        .unbox_reference::<sub::Binary>()
                        .byte_iter()
                        .collect();
                    let expected_returned_binary =
                        Term::slice_to_binary(&expected_returned_binary_bytes, &arc_process);

                    let start_length = Term::slice_to_tuple(&[start, length], &arc_process);

                    prop_assert_eq!(
                        erlang::binary_part_2(binary, start_length, &arc_process),
                        Ok(expected_returned_binary)
                    );

                    let returned_boxed =
                        erlang::binary_part_2(binary, start_length, &arc_process).unwrap();

                    prop_assert_eq!(returned_boxed.tag(), Boxed);

                    let returned_unboxed: &Term = returned_boxed.unbox_reference();

                    prop_assert_eq!(returned_unboxed.tag(), Subbinary);

                    let returned_subbinary: &sub::Binary = returned_boxed.unbox_reference();

                    prop_assert_eq!(
                        returned_subbinary.original,
                        binary.unbox_reference::<sub::Binary>().original
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_zero_start_and_byte_count_length_returns_subbinary_without_bit_count() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &binary::sub::with_size_range(
                    byte_offset(),
                    bit_offset(),
                    byte_count(),
                    (1_u8..=7_u8).boxed(),
                    arc_process.clone(),
                )
                .prop_map(|binary| {
                    (
                        binary,
                        0.into_process(&arc_process),
                        binary
                            .unbox_reference::<sub::Binary>()
                            .byte_count
                            .into_process(&arc_process),
                    )
                }),
                |(binary, start, length)| {
                    let expected_returned_binary_bytes: Vec<u8> = binary
                        .unbox_reference::<sub::Binary>()
                        .byte_iter()
                        .collect();
                    let expected_returned_binary =
                        Term::slice_to_binary(&expected_returned_binary_bytes, &arc_process);

                    let start_length = Term::slice_to_tuple(&[start, length], &arc_process);

                    prop_assert_eq!(
                        erlang::binary_part_2(binary, start_length, &arc_process),
                        Ok(expected_returned_binary)
                    );

                    let returned_boxed =
                        erlang::binary_part_2(binary, start_length, &arc_process).unwrap();

                    prop_assert_eq!(returned_boxed.tag(), Boxed);

                    let returned_unboxed: &Term = returned_boxed.unbox_reference();

                    prop_assert_eq!(returned_unboxed.tag(), Subbinary);

                    let returned_subbinary: &sub::Binary = returned_boxed.unbox_reference();

                    prop_assert_eq!(
                        returned_subbinary.original,
                        binary.unbox_reference::<sub::Binary>().original
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
