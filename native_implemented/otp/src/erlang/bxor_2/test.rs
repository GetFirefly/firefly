mod with_big_integer_left;
mod with_small_integer_left;

use num_bigint::BigInt;

use proptest::prop_assert_eq;
use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::Encoded;

use crate::erlang::bxor_2::result;
use crate::test::strategy;
use crate::test::with_process;

#[test]
fn without_integer_left_errors_badarith() {
    crate::test::without_integer_left_errors_badarith(file!(), result);
}

#[test]
fn without_integer_left_without_integer_right_errors_badarith() {
    crate::test::with_integer_left_without_integer_right_errors_badarith(file!(), result);
}

#[test]
fn with_same_integer_returns_zero() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, operand)| {
            prop_assert_eq!(
                result(&arc_process, operand, operand),
                Ok(arc_process.integer(0).unwrap())
            );

            Ok(())
        },
    );
}
