use super::*;

use std::convert::TryInto;

use liblumen_alloc::erts::term::prelude::{Encoded, SmallInteger};

#[test]
fn with_small_integer_returns_small_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::small::isize(),
            )
                .prop_flat_map(|(arc_process, integer)| {
                    let byte_vec = integer.to_string().as_bytes().to_owned();

                    (
                        Just(arc_process.clone()),
                        Just(integer),
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                    )
                })
        },
        |(arc_process, integer, binary)| {
            let result = result(&arc_process, binary);

            prop_assert!(result.is_ok());

            let term = result.unwrap();

            let small_integer_result: core::result::Result<SmallInteger, _> = term.try_into();

            prop_assert!(small_integer_result.is_ok());

            let small_integer = small_integer_result.unwrap();
            let small_integer_isize: isize = small_integer.into();

            prop_assert_eq!(small_integer_isize, integer);

            Ok(())
        },
    );
}

#[test]
fn with_big_integer_returns_big_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::big::isize(),
            )
                .prop_flat_map(|(arc_process, integer)| {
                    let byte_vec = integer.to_string().as_bytes().to_owned();

                    (
                        Just(arc_process.clone()),
                        Just(integer),
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                    )
                })
        },
        |(arc_process, integer, binary)| {
            let result = result(&arc_process, binary);

            prop_assert!(result.is_ok());

            let term = result.unwrap();

            prop_assert!(term.is_boxed_bigint());
            prop_assert_eq!(term, arc_process.integer(integer).unwrap());

            Ok(())
        },
    );
}

#[test]
fn with_non_decimal_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::containing_bytes(
                    "FF".as_bytes().to_owned(),
                    arc_process.clone(),
                ),
            )
        },
        |(arc_process, binary)| {
            prop_assert_badarg!(
                result(&arc_process, binary),
                format!("binary ({}) is not base 10", binary)
            );

            Ok(())
        },
    );
}
