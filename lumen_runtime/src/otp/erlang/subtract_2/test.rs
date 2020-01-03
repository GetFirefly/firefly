mod with_float_minuend;
mod with_integer_minuend;

use proptest::prop_assert;
use proptest::strategy::Just;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::subtract_2::native;
use crate::scheduler::with_process;
use crate::test::{run, strategy};

#[test]
fn without_number_minuend_errors_badarith() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_number(arc_process.clone()),
                strategy::term::is_number(arc_process.clone()),
            )
        },
        |(arc_process, minuend, subtrahend)| {
            prop_assert_badarith!(
                native(&arc_process, minuend, subtrahend),
                format!(
                    "minuend ({}) and subtrahend ({}) aren't both numbers",
                    minuend, subtrahend
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_number_minuend_without_number_subtrahend_errors_badarith() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_number(arc_process.clone()),
                strategy::term::is_not_number(arc_process.clone()),
            )
        },
        |(arc_process, minuend, subtrahend)| {
            prop_assert_badarith!(
                native(&arc_process, minuend, subtrahend),
                format!(
                    "minuend ({}) and subtrahend ({}) aren't both numbers",
                    minuend, subtrahend
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_integer_minuend_with_integer_subtrahend_returns_integer() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, minuend, subtrahend)| {
            let result = native(&arc_process, minuend, subtrahend);

            prop_assert!(result.is_ok());

            let difference = result.unwrap();

            prop_assert!(difference.is_integer());

            Ok(())
        },
    );
}

#[test]
fn with_integer_minuend_with_float_subtrahend_returns_float() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                strategy::term::float(arc_process.clone()),
            )
        },
        |(arc_process, minuend, subtrahend)| {
            let result = native(&arc_process, minuend, subtrahend);

            prop_assert!(result.is_ok());

            let difference = result.unwrap();

            prop_assert!(difference.is_float());

            Ok(())
        },
    );
}
