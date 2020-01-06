use super::*;

use crate::test::strategy::term::binary;
use crate::test::strategy::term::binary::sub::{bit_offset, byte_count, byte_offset};
use crate::test::strategy::NON_EMPTY_RANGE_INCLUSIVE;

#[test]
fn with_positive_start_and_positive_length_returns_subbinary() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                binary::sub::with_size_range(
                    byte_offset(),
                    bit_offset(),
                    (3_usize..=6_usize).boxed(),
                    (1_u8..=7_u8).boxed(),
                    arc_process.clone(),
                ),
            )
                .prop_flat_map(|(arc_process, binary)| {
                    let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();
                    let byte_count = subbinary.full_byte_len();

                    // `start` must be 2 less than `byte_count` so that `length` can be at least 1
                    // and still get a full byte
                    (
                        Just(arc_process.clone()),
                        Just(binary),
                        (1..=(byte_count - 2)),
                    )
                })
                .prop_flat_map(|(arc_process, binary, start)| {
                    let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();
                    let byte_count = subbinary.full_byte_len();

                    (
                        Just(arc_process.clone()),
                        Just(binary),
                        Just(start),
                        1..=(byte_count - start),
                    )
                })
                .prop_map(|(arc_process, binary, start, length)| {
                    let mut heap = arc_process.acquire_heap();

                    (
                        arc_process.clone(),
                        binary,
                        heap.integer(start).unwrap(),
                        heap.integer(length).unwrap(),
                    )
                })
        },
        |(arc_process, binary, start, length)| {
            let start_length = arc_process.tuple_from_slice(&[start, length]).unwrap();

            let result = native(&arc_process, binary.into(), start_length);

            prop_assert!(result.is_ok());

            let returned = result.unwrap();

            prop_assert!(returned.is_boxed_subbinary());

            Ok(())
        },
    );
}

#[test]
fn with_byte_count_start_and_negative_byte_count_length_returns_subbinary_without_bit_count() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                binary::sub::with_size_range(
                    byte_offset(),
                    bit_offset(),
                    NON_EMPTY_RANGE_INCLUSIVE.boxed(),
                    (1_u8..=7u8).boxed(),
                    arc_process.clone(),
                ),
            )
                .prop_map(|(arc_process, binary)| {
                    let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();
                    let byte_count = subbinary.full_byte_len();

                    let mut heap = arc_process.acquire_heap();

                    (
                        arc_process.clone(),
                        binary,
                        heap.integer(byte_count).unwrap(),
                        heap.integer(-(byte_count as isize)).unwrap(),
                    )
                })
        },
        |(arc_process, binary, start, length)| {
            let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();

            let expected_returned_binary_bytes: Vec<u8> = subbinary.full_byte_iter().collect();
            let expected_returned_binary = arc_process
                .binary_from_bytes(&expected_returned_binary_bytes)
                .unwrap();

            let start_length = arc_process.tuple_from_slice(&[start, length]).unwrap();

            prop_assert_eq!(
                native(&arc_process, binary, start_length),
                Ok(expected_returned_binary)
            );

            let returned = native(&arc_process, binary, start_length).unwrap();

            let returned_subbinary_result: core::result::Result<Boxed<SubBinary>, _> =
                returned.try_into();

            prop_assert!(returned_subbinary_result.is_ok());

            let returned_subbinary = returned_subbinary_result.unwrap();

            prop_assert_eq!(returned_subbinary.original(), subbinary.original());

            Ok(())
        },
    );
}

#[test]
fn with_zero_start_and_byte_count_length_returns_subbinary_without_bit_count() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                binary::sub::with_size_range(
                    byte_offset(),
                    bit_offset(),
                    byte_count(),
                    (1_u8..=7_u8).boxed(),
                    arc_process.clone(),
                ),
            )
                .prop_map(|(arc_process, binary)| {
                    let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();

                    (
                        arc_process.clone(),
                        binary,
                        arc_process.integer(0).unwrap(),
                        arc_process.integer(subbinary.full_byte_len()).unwrap(),
                    )
                })
        },
        |(arc_process, binary, start, length)| {
            let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();
            let expected_returned_binary_bytes: Vec<u8> = subbinary.full_byte_iter().collect();
            let expected_returned_binary = arc_process
                .binary_from_bytes(&expected_returned_binary_bytes)
                .unwrap();

            let start_length = arc_process.tuple_from_slice(&[start, length]).unwrap();

            prop_assert_eq!(
                native(&arc_process, binary, start_length),
                Ok(expected_returned_binary)
            );

            let returned = native(&arc_process, binary, start_length).unwrap();

            let returned_subbinary_result: core::result::Result<Boxed<SubBinary>, _> =
                returned.try_into();

            prop_assert!(returned_subbinary_result.is_ok());

            let returned_subbinary = returned_subbinary_result.unwrap();
            let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();

            prop_assert_eq!(returned_subbinary.original(), subbinary.original());

            Ok(())
        },
    );
}
