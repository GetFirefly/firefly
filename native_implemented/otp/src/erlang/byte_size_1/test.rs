use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};

use crate::erlang::byte_size_1::result;
use crate::test::strategy;

#[test]
fn without_bitstring_errors_badarg() {
    crate::test::without_bitstring_errors_badarg(file!(), result);
}

#[test]
fn with_heap_binary_is_byte_count() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), strategy::byte_vec()).prop_map(|(arc_process, byte_vec)| {
                (
                    arc_process.clone(),
                    byte_vec.len(),
                    arc_process.binary_from_bytes(&byte_vec).unwrap(),
                )
            })
        },
        |(arc_process, byte_count, bitstring)| {
            prop_assert_eq!(
                result(&arc_process, bitstring),
                Ok(arc_process.integer(byte_count).unwrap())
            );

            Ok(())
        },
    );
}

#[test]
fn with_subbinary_without_bit_count_is_byte_count() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), strategy::byte_vec()).prop_flat_map(
                |(arc_process, byte_vec)| {
                    (
                        Just(arc_process.clone()),
                        Just(byte_vec.len()),
                        strategy::term::binary::sub::containing_bytes(
                            byte_vec,
                            arc_process.clone(),
                        ),
                    )
                },
            )
        },
        |(arc_process, byte_count, bitstring)| {
            prop_assert_eq!(
                result(&arc_process, bitstring),
                Ok(arc_process.integer(byte_count).unwrap())
            );

            Ok(())
        },
    );
}

#[test]
fn with_subbinary_with_bit_count_is_byte_count_plus_one() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::sub::byte_count(),
            )
                .prop_flat_map(|(arc_process, byte_count)| {
                    (
                        Just(arc_process.clone()),
                        Just(byte_count),
                        strategy::term::binary::sub::with_size_range(
                            strategy::term::binary::sub::byte_offset(),
                            strategy::term::binary::sub::bit_offset(),
                            (byte_count..=byte_count).boxed(),
                            (1_u8..=7_u8).boxed(),
                            arc_process.clone(),
                        ),
                    )
                })
        },
        |(arc_process, byte_count, bitstring)| {
            prop_assert_eq!(
                result(&arc_process, bitstring),
                Ok(arc_process.integer(byte_count + 1).unwrap())
            );

            Ok(())
        },
    );
}
