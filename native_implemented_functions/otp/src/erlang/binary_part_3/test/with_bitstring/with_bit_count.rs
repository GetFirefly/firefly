use super::*;

use crate::test::strategy::term::binary;
use crate::test::strategy::term::binary::sub::{bit_offset, byte_offset};
use crate::test::strategy::NON_EMPTY_RANGE_INCLUSIVE;
use crate::test::{
    arc_process_subbinary_to_arc_process_subbinary_two_less_than_length_start,
    arc_process_to_arc_process_subbinary_zero_start_byte_count_length,
};

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
                .prop_flat_map(
                    arc_process_subbinary_to_arc_process_subbinary_two_less_than_length_start,
                )
                .prop_flat_map(|(arc_process, binary, start)| {
                    let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();

                    (
                        Just(arc_process.clone()),
                        Just(binary),
                        Just(start),
                        1..=(subbinary.full_byte_len() - start),
                    )
                })
                .prop_map(|(arc_process, binary, start, length)| {
                    (
                        arc_process.clone(),
                        binary,
                        arc_process.integer(start).unwrap(),
                        arc_process.integer(length).unwrap(),
                    )
                })
        },
        returns_subbinary,
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

                    (
                        arc_process.clone(),
                        binary,
                        arc_process.integer(byte_count).unwrap(),
                        arc_process.integer(-(byte_count as isize)).unwrap(),
                    )
                })
        },
        returns_subbinary_without_bit_count,
    );
}

#[test]
fn with_zero_start_and_byte_count_length_returns_subbinary_without_bit_count() {
    run!(
        arc_process_to_arc_process_subbinary_zero_start_byte_count_length,
        returns_subbinary_without_bit_count,
    );
}

fn returns_subbinary_without_bit_count(
    (arc_process, binary, start, length): (Arc<Process>, Term, Term, Term),
) -> TestCaseResult {
    let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();

    let expected_returned_binary_bytes: Vec<u8> = subbinary.full_byte_iter().collect();
    let expected_returned_binary = arc_process
        .binary_from_bytes(&expected_returned_binary_bytes)
        .unwrap();

    prop_assert_eq!(
        result(&arc_process, binary, start, length),
        Ok(expected_returned_binary)
    );

    let returned = result(&arc_process, binary, start, length).unwrap();

    let returned_subbinary_result: core::result::Result<Boxed<SubBinary>, _> = returned.try_into();

    prop_assert!(returned_subbinary_result.is_ok());

    let returned_subbinary = returned_subbinary_result.unwrap();
    let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();

    prop_assert_eq!(returned_subbinary.original(), subbinary.original());

    Ok(())
}
