mod with_function;

use std::convert::TryInto;

use anyhow::*;

use proptest::prop_assert_eq;
use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::atom;
use liblumen_alloc::erts::process::Status;

use crate::otp::erlang::spawn_monitor_1::native;
use crate::registry::pid_to_process;
use crate::test::strategy::term::function;
use crate::test::strategy;

#[test]
fn without_function_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_function(arc_process.clone()),
            )
        },
        |(arc_process, function)| {
            prop_assert_badarg!(
                native(&arc_process, function),
                format!("function ({}) is not a function", function)
            );

            Ok(())
        },
    );
}
