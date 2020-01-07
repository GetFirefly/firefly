mod with_big_integer_left;
mod with_small_integer_left;

use proptest::strategy::Just;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::Encoded;

use crate::otp::erlang::band_2::native;
use crate::scheduler::with_process;
use crate::test::{count_ones, strategy};

#[test]
fn without_integer_right_errors_badarith() {
    crate::test::with_integer_left_without_integer_right_errors_badarith(file!(), native);
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
            prop_assert_eq!(native(&arc_process, operand, operand), Ok(operand));

            Ok(())
        },
    );
}
