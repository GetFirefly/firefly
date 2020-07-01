mod with_big_integer_augend;
mod with_float_augend;
mod with_small_integer_augend;

use std::sync::Arc;

use proptest::arbitrary::any;
use proptest::strategy::{Just, Strategy};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::add_2::result;
use crate::test::{run, strategy, with_process};

#[test]
fn without_number_augend_errors_badarith() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_number(arc_process.clone()),
                strategy::term::is_number(arc_process.clone()),
            )
        },
        |(arc_process, augend, addend)| {
            prop_assert_badarith!(
                result(&arc_process, augend, addend),
                format!(
                    "augend ({}) and addend ({}) aren't both numbers",
                    augend, addend
                )
            );

            Ok(())
        },
    );
}

fn without_number_addend_errors_badarith<F, S>(source_file: &'static str, augend_strategy: F)
where
    F: Fn(Arc<Process>) -> S,
    S: Strategy<Value = Term>,
{
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                augend_strategy(arc_process.clone()),
                strategy::term::is_not_number(arc_process),
            )
        },
        |(arc_process, augend, addend)| {
            prop_assert_badarith!(
                result(&arc_process, augend, addend),
                format!(
                    "augend ({}) and addend ({}) aren't both numbers",
                    augend, addend
                )
            );

            Ok(())
        },
    );
}
