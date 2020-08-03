use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_less_than_byte_len_returns_binary_prefix_and_suffix_binary() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), 2_usize..=4_usize).prop_flat_map(
                |(arc_process, byte_len)| {
                    (
                        Just(arc_process.clone()),
                        strategy::byte_vec::with_size_range((byte_len..=byte_len).into()),
                        1..byte_len,
                    )
                },
            )
        },
        |(arc_process, byte_vec, index)| {
            let binary = arc_process.binary_from_bytes(&byte_vec).unwrap();
            let position = arc_process.integer(index).unwrap();

            let prefix_bytes = &byte_vec[0..index];
            let prefix = arc_process.binary_from_bytes(prefix_bytes).unwrap();

            let suffix_bytes = &byte_vec[index..];
            let suffix = arc_process.binary_from_bytes(suffix_bytes).unwrap();

            prop_assert_eq!(
                result(&arc_process, binary, position),
                Ok(arc_process.tuple_from_slice(&[prefix, suffix]).unwrap())
            );

            Ok(())
        },
    );
}

#[test]
fn with_byte_len_returns_subbinary_and_empty_suffix() {
    run!(
        |arc_process| (Just(arc_process.clone()), strategy::byte_vec()),
        |(arc_process, byte_vec)| {
            let binary = arc_process.binary_from_bytes(&byte_vec).unwrap();
            let position = arc_process.integer(byte_vec.len()).unwrap();

            prop_assert_eq!(
                result(&arc_process, binary, position),
                Ok(arc_process
                    .tuple_from_slice(&[binary, arc_process.binary_from_bytes(&[]).unwrap()],)
                    .unwrap())
            );

            Ok(())
        },
    );
}

#[test]
fn with_greater_than_byte_len_errors_badarg() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), strategy::byte_vec()).prop_flat_map(
                |(arc_process, byte_vec)| {
                    let min = byte_vec.len() + 1;
                    let max = std::isize::MAX as usize;

                    (Just(arc_process.clone()), Just(byte_vec), min..=max)
                },
            )
        },
        |(arc_process, byte_vec, index)| {
            let binary = arc_process.binary_from_bytes(&byte_vec).unwrap();
            let position = arc_process.integer(index).unwrap();

            prop_assert_badarg!(
                result(&arc_process, binary, position),
                format!(
                    "index ({}) exceeds full byte length ({})",
                    index,
                    byte_vec.len()
                )
            );

            Ok(())
        },
    );
}
