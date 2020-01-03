mod with_big_integer_integer;
mod with_small_integer_integer;

use num_bigint::BigInt;

use proptest::prop_assert_eq;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::term::prelude::Encoded;

use crate::otp::erlang;
use crate::otp::erlang::bsl_2::native;
use crate::scheduler::with_process;
use crate::test::{run, strategy};

#[test]
fn without_integer_integer_errors_badarith() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, integer, shift)| {
            prop_assert_badarith!(
                native(&arc_process, integer, shift),
                format!("integer ({}) is not an integer", integer)
            );

            Ok(())
        },
    );
}

#[test]
fn with_integer_integer_without_integer_shift_errors_badarith() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
            )
        },
        |(arc_process, integer, shift)| {
            prop_assert_badarith!(
                native(&arc_process, integer, shift),
                format!("shift ({}) is not an integer", shift)
            );

            Ok(())
        },
    );
}

#[test]
fn with_integer_integer_with_zero_shift_returns_same_integer() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, integer)| {
            let shift = arc_process.integer(0).unwrap();

            prop_assert_eq!(native(&arc_process, integer, shift), Ok(integer));

            Ok(())
        },
    );
}

#[test]
fn with_integer_integer_with_integer_shift_is_the_same_as_bsr_with_negated_shift() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                shift(),
            )
        },
        |(arc_process, integer, shift)| {
            let negated_shift = -1 * shift;

            prop_assert_eq!(
                native(
                    &arc_process,
                    integer,
                    arc_process.integer(shift as isize).unwrap(),
                ),
                erlang::bsr_2::native(
                    &arc_process,
                    integer,
                    arc_process.integer(negated_shift as isize).unwrap(),
                )
            );

            Ok(())
        },
    );
}

fn shift() -> BoxedStrategy<i8> {
    // any::<i8> is not symmetric because i8::MIN is -128 while i8::MAX is 127, so make symmetric
    // range
    (-127_i8..=127_i8).boxed()
}
