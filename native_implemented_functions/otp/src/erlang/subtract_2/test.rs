mod with_float_minuend;
mod with_integer_minuend;

use proptest::prop_assert;
use proptest::strategy::Just;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::subtract_2::result;
use crate::test::strategy;
use crate::test::with_process;

#[test]
fn without_number_minuend_errors_badarith() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_number(arc_process.clone()),
                strategy::term::is_number(arc_process.clone()),
            )
        },
        |(arc_process, minuend, subtrahend)| {
            prop_assert_badarith!(
                result(&arc_process, minuend, subtrahend),
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
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_number(arc_process.clone()),
                strategy::term::is_not_number(arc_process.clone()),
            )
        },
        |(arc_process, minuend, subtrahend)| {
            prop_assert_badarith!(
                result(&arc_process, minuend, subtrahend),
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
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, minuend, subtrahend)| {
            let result = result(&arc_process, minuend, subtrahend);

            prop_assert!(result.is_ok());

            let difference = result.unwrap();

            prop_assert!(difference.is_integer());

            Ok(())
        },
    );
}

#[test]
fn with_integer_minuend_with_float_subtrahend_returns_float() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                strategy::term::float(arc_process.clone()),
            )
        },
        |(arc_process, minuend, subtrahend)| {
            let result = result(&arc_process, minuend, subtrahend);

            prop_assert!(result.is_ok());

            let difference = result.unwrap();

            prop_assert!(difference.is_float());

            Ok(())
        },
    );
}
