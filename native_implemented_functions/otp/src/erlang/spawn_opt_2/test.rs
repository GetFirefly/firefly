mod with_function;

use std::convert::TryInto;
use std::sync::Arc;

use anyhow::*;

use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{atom, exit};

use crate::runtime::registry::pid_to_process;
use crate::runtime::scheduler;

use crate::erlang::spawn_opt_2::native;
use crate::test;
use crate::test::strategy::term::function;
use crate::test::{badarity_reason, has_message, prop_assert_exits_badarity, strategy};

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
            let options = Term::NIL;

            prop_assert_badarg!(
                native(&arc_process, function, options),
                format!("function ({}) is not a function", function)
            );

            Ok(())
        },
    );
}
