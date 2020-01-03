mod with_function;

use std::convert::TryInto;

use proptest::prop_assert_eq;
use proptest::strategy::Just;

use liblumen_alloc::erts::process::Status;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::spawn_1::native;
use crate::registry::pid_to_process;
use crate::test::strategy::term::function;
use crate::test::{prop_assert_exits_badarity, run, strategy};

#[test]
fn without_function_errors_badarg() {
    run(
        file!(),
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
