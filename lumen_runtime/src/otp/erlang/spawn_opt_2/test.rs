mod with_function;

use std::convert::TryInto;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::process::Status;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{atom, badarg, badarity};

use crate::otp::erlang::spawn_opt_2::native;
use crate::registry::pid_to_process;
use crate::scheduler::{with_process_arc, Scheduler};
use crate::test::strategy;
use crate::test::strategy::term::function;

#[test]
fn without_function_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_function(arc_process.clone()),
                |function| {
                    let options = Term::NIL;

                    prop_assert_eq!(
                        native(&arc_process, function, options),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
