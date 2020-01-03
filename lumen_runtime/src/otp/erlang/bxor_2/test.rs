mod with_big_integer_left;
mod with_small_integer_left;

use num_bigint::BigInt;

use proptest::prop_assert_eq;
use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::Encoded;

use crate::otp::erlang::bxor_2::native;
use crate::scheduler::with_process;
use crate::test::{run, strategy};

#[test]
fn without_integer_left_errors_badarith() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
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
fn without_integer_left_without_integer_right_errors_badarith() {
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
fn with_same_integer_returns_zero() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, operand)| {
            prop_assert_eq!(
                native(&arc_process, operand, operand),
                Ok(arc_process.integer(0).unwrap())
            );

            Ok(())
        },
    );
}
