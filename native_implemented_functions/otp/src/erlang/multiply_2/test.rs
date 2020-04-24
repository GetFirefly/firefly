mod with_big_integer_multiplier;
mod with_float_multiplier;
mod with_small_integer_multiplier;

use proptest::prop_assert;
use proptest::strategy::Just;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::multiply_2::result;
use crate::test::strategy;
use crate::test::with_process;

#[test]
fn without_number_multiplier_errors_badarith() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_number(arc_process.clone()),
                strategy::term::is_number(arc_process.clone()),
            )
        },
        |(arc_process, multiplier, multiplicand)| {
            prop_assert_badarith!(
                result(&arc_process, multiplier, multiplicand),
                format!(
                    "multiplier ({}) and multiplicand ({}) aren't both numbers",
                    multiplier, multiplicand
                )
            );

            Ok(())
        },
    );
}
