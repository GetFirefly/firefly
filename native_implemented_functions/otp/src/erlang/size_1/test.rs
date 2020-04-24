use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::size_1::result;
use crate::test::strategy;

#[test]
fn without_tuple_or_bitstring_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Term must not be a tuple or bitstring", |term| {
                        !(term.is_boxed_tuple() || term.is_bitstring())
                    }),
            )
        },
        |(arc_process, binary_or_tuple)| {
            prop_assert_badarg!(
                result(&arc_process, binary_or_tuple),
                format!(
                    "binary_or_tuple ({}) is neither a binary nor a tuple",
                    binary_or_tuple
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_tuple_returns_arity() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), 0_usize..=3_usize).prop_flat_map(|(arc_process, size)| {
                (
                    Just(arc_process.clone()),
                    Just(size),
                    strategy::term::tuple::intermediate(
                        strategy::term(arc_process.clone()),
                        (size..=size).into(),
                        arc_process.clone(),
                    ),
                )
            })
        },
        |(arc_process, size, term)| {
            prop_assert_eq!(
                result(&arc_process, term),
                Ok(arc_process.integer(size).unwrap())
            );

            Ok(())
        },
    );
}

#[test]
fn with_bitstring_is_byte_len() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_bitstring(arc_process.clone()),
            )
        },
        |(arc_process, term)| {
            let full_byte_len = match term.decode().unwrap() {
                TypedTerm::HeapBinary(heap_binary) => heap_binary.full_byte_len(),
                TypedTerm::SubBinary(subbinary) => subbinary.full_byte_len(),
                _ => unreachable!(),
            };

            prop_assert_eq!(
                result(&arc_process, term),
                Ok(arc_process.integer(full_byte_len).unwrap())
            );

            Ok(())
        },
    );
}
