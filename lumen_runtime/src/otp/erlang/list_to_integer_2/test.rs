use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use radix_fmt::radix;

use crate::otp::erlang::list_to_integer_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_list(arc_process.clone()),
                    strategy::term::is_base(arc_process.clone()),
                ),
                |(list, base)| {
                    prop_assert_badarg!(
                        native(&arc_process, list, base),
                        format!("list ({}) is not a list", list)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_list_without_base_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    any::<isize>()
                        .prop_map(|i| arc_process.charlist_from_str(&i.to_string()).unwrap()),
                    strategy::term::is_not_base(arc_process.clone()),
                ),
                |(list, base)| {
                    prop_assert_badarg!(
                        native(&arc_process, list, base),
                        "base must be an integer in 2-36"
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_list_with_integer_in_base_returns_integers() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(any::<isize>(), strategy::base::base()).prop_map(|(integer, base)| {
                    // `radix` does 2's complement for negatives, but that's not what Erlang expects
                    let string = if integer < 0 {
                        format!("-{}", radix(-1 * integer, base))
                    } else {
                        format!("{}", radix(integer, base))
                    };

                    (
                        integer,
                        arc_process.charlist_from_str(&string).unwrap(),
                        arc_process.integer(base).unwrap(),
                    )
                }),
                |(integer, list, base)| {
                    prop_assert_eq!(
                        native(&arc_process, list, base),
                        Ok(arc_process.integer(integer).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_list_without_integer_in_base_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::base::base().prop_map(|base| {
                    let invalid_digit = match base {
                        2..=9 => b'0' + base,
                        10..=36 => b'A' + (base - 10),
                        _ => unreachable!(),
                    };

                    let string = String::from_utf8(vec![invalid_digit]).unwrap();

                    (
                        string.clone(),
                        arc_process.charlist_from_str(&string).unwrap(),
                        arc_process.integer(base).unwrap(),
                    )
                }),
                |(string, list, base)| {
                    prop_assert_badarg!(
                        native(&arc_process, list, base),
                        format!("string ({}) is not in base ({})", string, base)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
