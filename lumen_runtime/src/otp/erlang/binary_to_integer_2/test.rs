use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use radix_fmt::radix;

use liblumen_alloc::badarg;

use crate::otp::erlang::binary_to_integer_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_binary_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_binary(arc_process.clone()),
                    strategy::term::is_base(arc_process.clone()),
                ),
                |(binary, base)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, base),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_without_base_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_binary(arc_process.clone()),
                    strategy::term::is_not_base(arc_process.clone()),
                ),
                |(binary, base)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, base),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_with_integer_in_base_returns_integers() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(any::<isize>(), strategy::base::base()).prop_flat_map(|(integer, base)| {
                    // `radix` does 2's complement for negatives, but that's not what Erlang expects
                    let string = if integer < 0 {
                        format!("-{}", radix(-1 * integer, base))
                    } else {
                        format!("{}", radix(integer, base))
                    };

                    let byte_vec = string.as_bytes().to_owned();

                    (
                        Just(integer),
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                        Just(arc_process.integer(base).unwrap()),
                    )
                }),
                |(integer, binary, base)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, base),
                        Ok(arc_process.integer(integer).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_without_integer_in_base_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::base::base().prop_flat_map(|base| {
                    let invalid_digit = match base {
                        2..=9 => b'0' + base,
                        10..=36 => b'A' + (base - 10),
                        _ => unreachable!(),
                    };

                    let byte_vec = vec![invalid_digit];

                    (
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                        Just(arc_process.integer(base).unwrap()),
                    )
                }),
                |(binary, base)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, base),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
