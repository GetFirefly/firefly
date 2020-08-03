use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};

use radix_fmt::radix;

use crate::erlang::list_to_integer_2::result;
use crate::test::strategy;

#[test]
fn without_list_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
                strategy::term::is_base(arc_process.clone()),
            )
        },
        |(arc_process, list, base)| {
            prop_assert_badarg!(
                result(&arc_process, list, base),
                format!("list ({}) is not a list", list)
            );

            Ok(())
        },
    );
}

#[test]
fn with_list_without_base_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                (Just(arc_process.clone()), any::<isize>()).prop_map(|(arc_process, i)| {
                    arc_process.charlist_from_str(&i.to_string()).unwrap()
                }),
                strategy::term::is_not_base(arc_process.clone()),
            )
        },
        |(arc_process, list, base)| {
            prop_assert_badarg!(
                result(&arc_process, list, base),
                "base must be an integer in 2-36"
            );

            Ok(())
        },
    );
}

#[test]
fn with_list_with_integer_in_base_returns_integers() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                any::<isize>(),
                strategy::base::base(),
            )
                .prop_map(|(arc_process, integer, base)| {
                    // `radix` does 2's complement for negatives, but that's not what Erlang expects
                    let string = if integer < 0 {
                        format!("-{}", radix(-1 * integer, base))
                    } else {
                        format!("{}", radix(integer, base))
                    };

                    (
                        arc_process.clone(),
                        integer,
                        arc_process.charlist_from_str(&string).unwrap(),
                        arc_process.integer(base).unwrap(),
                    )
                })
        },
        |(arc_process, integer, list, base)| {
            prop_assert_eq!(
                result(&arc_process, list, base),
                Ok(arc_process.integer(integer).unwrap())
            );

            Ok(())
        },
    );
}

#[test]
fn with_list_without_integer_in_base_errors_badarg() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), strategy::base::base()).prop_map(|(arc_process, base)| {
                let invalid_digit = match base {
                    2..=9 => b'0' + base,
                    10..=36 => b'A' + (base - 10),
                    _ => unreachable!(),
                };

                let string = String::from_utf8(vec![invalid_digit]).unwrap();

                (
                    arc_process.clone(),
                    string.clone(),
                    arc_process.charlist_from_str(&string).unwrap(),
                    arc_process.integer(base).unwrap(),
                )
            })
        },
        |(arc_process, string, list, base)| {
            prop_assert_badarg!(
                result(&arc_process, list, base),
                format!("list ('{}') is not in base ({})", string, base)
            );

            Ok(())
        },
    );
}
