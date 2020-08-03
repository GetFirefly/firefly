use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};

use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::binary_to_list_1::result;
use crate::test::strategy;

#[test]
fn without_binary_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_binary(arc_process.clone()),
            )
        },
        |(arc_process, binary)| {
            prop_assert_badarg!(result(&arc_process, binary), format!("binary ({})", binary));

            Ok(())
        },
    );
}

#[test]
fn with_binary_returns_list_of_bytes() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), strategy::byte_vec()).prop_flat_map(
                |(arc_process, byte_vec)| {
                    (
                        Just(arc_process.clone()),
                        Just(byte_vec.clone()),
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                    )
                },
            )
        },
        |(arc_process, byte_vec, binary)| {
            // not using an iterator because that would too closely match the code under
            // test
            let list = match byte_vec.len() {
                0 => Term::NIL,
                1 => arc_process
                    .cons(arc_process.integer(byte_vec[0]).unwrap(), Term::NIL)
                    .unwrap(),
                2 => arc_process
                    .cons(
                        arc_process.integer(byte_vec[0]).unwrap(),
                        arc_process
                            .cons(arc_process.integer(byte_vec[1]).unwrap(), Term::NIL)
                            .unwrap(),
                    )
                    .unwrap(),
                3 => arc_process
                    .cons(
                        arc_process.integer(byte_vec[0]).unwrap(),
                        arc_process
                            .cons(
                                arc_process.integer(byte_vec[1]).unwrap(),
                                arc_process
                                    .cons(arc_process.integer(byte_vec[2]).unwrap(), Term::NIL)
                                    .unwrap(),
                            )
                            .unwrap(),
                    )
                    .unwrap(),
                len => unimplemented!("len = {:?}", len),
            };

            prop_assert_eq!(result(&arc_process, binary), Ok(list));

            Ok(())
        },
    );
}
