use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_binary_right_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::binary::heap(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Right must not be a binary", |v| !v.is_binary()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_heap_binary_right_returns_true() {
    run!(
        |arc_process| strategy::term::binary::heap(arc_process.clone()),
        |operand| {
            prop_assert_eq!(result(operand, operand), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_value_heap_binary_right_returns_true() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), strategy::byte_vec()).prop_map(|(arc_process, byte_vec)| {
                let mut heap = arc_process.acquire_heap();

                (
                    heap.binary_from_bytes(&byte_vec).unwrap(),
                    heap.binary_from_bytes(&byte_vec).unwrap(),
                )
            })
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_heap_binary_right_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::binary::heap(arc_process.clone()),
                strategy::term::binary::heap(arc_process.clone()),
            )
                .prop_filter("Heap binaries must be different", |(left, right)| {
                    left != right
                })
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_subbinary_right_with_same_bytes_returns_true() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::sub::is_binary(arc_process.clone()),
            )
                .prop_map(|(arc_process, subbinary_term)| {
                    let subbinary: Boxed<SubBinary> = subbinary_term.try_into().unwrap();
                    let heap_binary_byte_vec: Vec<u8> = subbinary.full_byte_iter().collect();

                    let heap_binary = arc_process
                        .binary_from_bytes(&heap_binary_byte_vec)
                        .unwrap();
                    (heap_binary, subbinary_term)
                })
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_subbinary_right_with_different_bytes_returns_false() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::sub::is_binary::is_not_empty(arc_process.clone()),
            )
                .prop_map(|(arc_process, subbinary_term)| {
                    let subbinary: Boxed<SubBinary> = subbinary_term.try_into().unwrap();
                    // same size, but different values by inverting
                    let heap_binary_byte_vec: Vec<u8> =
                        subbinary.full_byte_iter().map(|b| !b).collect();

                    let heap_binary = arc_process
                        .binary_from_bytes(&heap_binary_byte_vec)
                        .unwrap();
                    (heap_binary, subbinary_term)
                })
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}
