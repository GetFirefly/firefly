mod with_big_integer_left;
mod with_small_integer_left;

use proptest::strategy::Just;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::Encoded;

use crate::otp::erlang::band_2::native;
use crate::scheduler::with_process;
use crate::test::{count_ones, run, strategy};

#[test]
fn without_integer_right_errors_badarith() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
            )
        },
        |(arc_process, left, right)| {
            prop_assert_badarith!(
                native(&arc_process, left, right),
                format!(
                    "left_integer ({}) and right_integer ({}) are not both integers",
                    left, right
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_same_integer_returns_same_integer() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, operand)| {
            prop_assert_eq!(native(&arc_process, operand, operand), Ok(operand));

            Ok(())
        },
    );
}
