mod with_big_integer_augend;
mod with_float_augend;
mod with_small_integer_augend;

use proptest::arbitrary::any;
use proptest::strategy::Just;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::add_2::native;
use crate::scheduler::with_process;
use crate::test::{run, strategy};

#[test]
fn without_number_augend_errors_badarith() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_number(arc_process.clone()),
                strategy::term::is_number(arc_process.clone()),
            )
        },
        |(arc_process, augend, addend)| {
            prop_assert_badarith!(
                native(&arc_process, augend, addend),
                format!(
                    "augend ({}) and addend ({}) aren't both numbers",
                    augend, addend
                )
            );

            Ok(())
        },
    );
}
