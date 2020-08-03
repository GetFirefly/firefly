use num_bigint::BigInt;

use num_traits::Num;

use proptest::prop_assert;
use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::{Encoded, TypedTerm};

use crate::erlang::bnot_1::result;
use crate::test::strategy;
use crate::test::with_process;

#[test]
fn without_integer_errors_badarith() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
            )
        },
        |(arc_process, integer)| {
            prop_assert_badarith!(
                result(&arc_process, integer),
                format!("integer ({}) is not an integer", integer)
            );

            Ok(())
        },
    );
}

#[test]
fn with_small_integer_returns_small_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::small(arc_process.clone()),
            )
        },
        |(arc_process, operand)| {
            let result = result(&arc_process, operand);

            prop_assert!(result.is_ok());

            let inverted = result.unwrap();

            prop_assert!(inverted.is_smallint());

            Ok(())
        },
    );
}

#[test]
fn with_small_integer_inverts_bits() {
    with_process(|process| {
        let integer = process.integer(0b10).unwrap();

        assert_eq!(result(&process, integer), Ok(process.integer(-3).unwrap()))
    });
}

#[test]
fn with_big_integer_inverts_bits() {
    with_process(|process| {
        let integer_big_int = <BigInt as Num>::from_str_radix(
            "1010101010101010101010101010101010101010101010101010101010101010",
            2,
        )
        .unwrap();
        let integer = process.integer(integer_big_int).unwrap();

        assert!(integer.is_boxed_bigint());

        assert_eq!(
            result(&process, integer),
            Ok(process
                .integer(<BigInt as Num>::from_str_radix("-12297829382473034411", 10,).unwrap())
                .unwrap())
        );
    });
}

#[test]
fn with_big_integer_returns_big_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::big(arc_process.clone()),
            )
        },
        |(arc_process, operand)| {
            let result = result(&arc_process, operand);

            prop_assert!(result.is_ok());

            let inverted = result.unwrap();

            match inverted.decode().unwrap() {
                TypedTerm::BigInteger(_) => prop_assert!(true),
                _ => prop_assert!(false),
            }

            Ok(())
        },
    );
}
