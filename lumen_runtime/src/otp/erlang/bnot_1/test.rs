use num_bigint::BigInt;

use num_traits::Num;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarith;
use liblumen_alloc::erts::term::prelude::TypedTerm;

use crate::otp::erlang::bnot_1::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;

#[test]
fn without_integer_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_integer(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(native(&arc_process, operand), Err(badarith!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_small_integer_returns_small_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::small(arc_process.clone()),
                |operand| {
                    let result = native(&arc_process, operand);

                    prop_assert!(result.is_ok());

                    let inverted = result.unwrap();

                    prop_assert!(inverted.is_smallint());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_small_integer_inverts_bits() {
    with_process(|process| {
        let integer = process.integer(0b10).unwrap();

        assert_eq!(native(&process, integer), Ok(process.integer(-3).unwrap()))
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

        assert!(integer.is_bigint());

        assert_eq!(
            native(&process, integer),
            Ok(process
                .integer(<BigInt as Num>::from_str_radix("-12297829382473034411", 10,).unwrap())
                .unwrap())
        );
    });
}

#[test]
fn with_big_integer_returns_big_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::big(arc_process.clone()),
                |operand| {
                    let result = native(&arc_process, operand);

                    prop_assert!(result.is_ok());

                    let inverted = result.unwrap();

                    match inverted.to_typed_term().unwrap() {
                        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                            TypedTerm::BigInteger(_) => prop_assert!(true),
                            _ => prop_assert!(false),
                        },
                        _ => prop_assert!(false),
                    }

                    Ok(())
                },
            )
            .unwrap();
    });
}
