mod with_big_integer_dividend;
mod with_small_integer_dividend;

use proptest::prop_assert_eq;
use proptest::strategy::{BoxedStrategy, Just};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang;
use crate::otp::erlang::div_2::native;
use crate::scheduler::with_process;
use crate::test::{run, strategy};

#[test]
fn without_integer_dividend_errors_badarith() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, dividend, divisor)| {
            prop_assert_badarith!(
                native(&arc_process, dividend, divisor),
                format!(
                    "dividend ({}) and divisor ({}) are not both numbers",
                    dividend, divisor
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_integer_dividend_without_integer_divisor_errors_badarith() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
            )
        },
        |(arc_process, dividend, divisor)| {
            prop_assert_badarith!(
                native(&arc_process, dividend, divisor),
                format!(
                    "dividend ({}) and divisor ({}) are not both numbers",
                    dividend, divisor
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_integer_dividend_with_zero_divisor_errors_badarith() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                Just(arc_process.integer(0).unwrap()),
            )
        },
        |(arc_process, dividend, divisor)| {
            prop_assert_badarith!(
                native(&arc_process, dividend, divisor),
                format!("divisor ({}) cannot be zero", divisor)
            );

            Ok(())
        },
    );
}
