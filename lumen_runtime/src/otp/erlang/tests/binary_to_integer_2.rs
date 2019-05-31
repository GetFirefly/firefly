use super::*;

use proptest::strategy::Strategy;
use radix_fmt::radix;

#[test]
fn without_binary_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_binary(arc_process.clone()),
                    term_is_base(arc_process.clone()),
                ),
                |(binary, base)| {
                    prop_assert_eq!(
                        erlang::binary_to_integer_2(binary, base, &arc_process),
                        Err(badarg!())
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
                    term_is_not_base(arc_process.clone()),
                ),
                |(binary, base)| {
                    prop_assert_eq!(
                        erlang::binary_to_integer_2(binary, base, &arc_process),
                        Err(badarg!())
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
                &(any::<isize>(), base()).prop_flat_map(|(integer, base)| {
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
                        Just(base.into_process(&arc_process)),
                    )
                }),
                |(integer, binary, base)| {
                    prop_assert_eq!(
                        erlang::binary_to_integer_2(binary, base, &arc_process),
                        Ok(integer.into_process(&arc_process))
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
                &base().prop_flat_map(|base| {
                    let invalid_digit = match base {
                        2..=9 => b'0' + base,
                        10..=36 => b'A' + (base - 10),
                        _ => unreachable!(),
                    };

                    let byte_vec = vec![invalid_digit];

                    (
                        strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                        Just(base.into_process(&arc_process)),
                    )
                }),
                |(binary, base)| {
                    prop_assert_eq!(
                        erlang::binary_to_integer_2(binary, base, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn base() -> impl Strategy<Value = u8> {
    (2_u8..=36_u8).boxed()
}

fn term_is_base(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    base().prop_map(move |base| base.into_process(&arc_process))
}

fn term_is_not_base(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    strategy::term(arc_process).prop_filter("Cannot be a base (2-36)", |term| match term.tag() {
        SmallInteger => {
            let integer: isize = unsafe { term.small_integer_to_isize() };

            (2 <= integer) && (integer <= 36)
        }
        _ => true,
    })
}
