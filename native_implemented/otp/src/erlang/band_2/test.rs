mod with_big_integer_left;
mod with_small_integer_left;

use std::sync::Arc;

use proptest::strategy::{Just, Strategy};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::band_2::result;
use crate::test::with_process;
use crate::test::{count_ones, run, strategy};

#[test]
fn without_integer_right_errors_badarith() {
    crate::test::with_integer_left_without_integer_right_errors_badarith(file!(), result);
}

#[test]
fn with_same_integer_returns_same_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, operand)| {
            prop_assert_eq!(result(&arc_process, operand, operand), Ok(operand));

            Ok(())
        },
    );
}

fn with_integer_right_returns_bitwise_and<F, S>(source_file: &'static str, left_strategy: F)
where
    F: Fn(Arc<Process>) -> S,
    S: Strategy<Value = Term>,
{
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                left_strategy(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, left, right)| {
            let result = result(&arc_process, left, right);

            prop_assert!(result.is_ok());

            let band = result.unwrap();

            prop_assert!(band.is_integer());
            prop_assert!(count_ones(band) <= count_ones(left));
            prop_assert!(count_ones(band) <= count_ones(right));

            Ok(())
        },
    );
}
