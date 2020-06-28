use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};

use radix_fmt::radix;

use crate::erlang::binary_to_integer_2::result;
use crate::test::strategy;

#[test]
fn without_binary_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_binary(arc_process.clone()),
                strategy::term::is_base(arc_process.clone()),
            )
        },
        |(arc_process, binary, base)| {
            prop_assert_badarg!(
                result(&arc_process, binary, base),
                format!("binary ({}) must be a binary", binary)
            );

            Ok(())
        },
    );
}

#[test]
fn with_utf8_binary_without_base_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::is_utf8(arc_process.clone()),
                strategy::term::is_not_base(arc_process.clone()),
            )
        },
        |(arc_process, binary, base)| {
            prop_assert_badarg!(
                result(&arc_process, binary, base),
                format!("base must be an integer in 2-36")
            );

            Ok(())
        },
    );
}

#[test]
fn with_binary_with_integer_in_base_returns_integers() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                any::<isize>(),
                strategy::base::base(),
            )
                .prop_flat_map(|(arc_process, integer, base)| {
                    // `radix` does 2's complement for negatives, but that's not what Erlang expects
                    let string = if integer < 0 {
                        format!("-{}", radix(-1 * integer, base))
                    } else {
                        format!("{}", radix(integer, base))
                    };

                    let byte_vec = string.as_bytes().to_owned();

                    (
                        Just(arc_process.clone()),
                        Just(integer),
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                        Just(arc_process.integer(base).unwrap()),
                    )
                })
        },
        |(arc_process, integer, binary, base)| {
            prop_assert_eq!(
                result(&arc_process, binary, base),
                Ok(arc_process.integer(integer).unwrap())
            );

            Ok(())
        },
    );
}

#[test]
fn with_binary_without_integer_in_base_errors_badarg() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), strategy::base::base()).prop_flat_map(
                |(arc_process, base)| {
                    let invalid_digit = match base {
                        2..=9 => b'0' + base,
                        10..=36 => b'A' + (base - 10),
                        _ => unreachable!(),
                    };

                    let byte_vec = vec![invalid_digit];

                    (
                        Just(arc_process.clone()),
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                        Just(arc_process.integer(base).unwrap()),
                    )
                },
            )
        },
        |(arc_process, binary, base)| {
            prop_assert_badarg!(
                result(&arc_process, binary, base),
                format!("binary ({}) is not in base ({})", binary, base)
            );

            Ok(())
        },
    );
}
