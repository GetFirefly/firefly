mod with_big_integer_dividend;
mod with_small_integer_dividend;

use proptest::prop_assert_eq;
use proptest::strategy::{BoxedStrategy, Just};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarith;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang;
use crate::otp::erlang::div_2::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;

#[test]
fn without_integer_dividend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_integer(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        native(&arc_process, dividend, divisor),
                        Err(badarith!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_dividend_without_integer_divisor_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        native(&arc_process, dividend, divisor),
                        Err(badarith!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_dividend_with_zero_divisor_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    Just(arc_process.integer(0).unwrap()),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        native(&arc_process, dividend, divisor),
                        Err(badarith!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
