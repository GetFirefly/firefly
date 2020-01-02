mod with_function;

use std::convert::TryInto;

use anyhow::*;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::atom;
use liblumen_alloc::erts::process::Status;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::spawn_opt_2::native;
use crate::registry::pid_to_process;
use crate::scheduler::{with_process_arc, Scheduler};
use crate::test::strategy::term::function;
use crate::test::{prop_assert_exits_badarity, strategy};

#[test]
fn without_function_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_function(arc_process.clone()),
                |function| {
                    let options = Term::NIL;

                    prop_assert_badarg!(
                        native(&arc_process, function, options),
                        format!("function ({}) is not a function", function)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
