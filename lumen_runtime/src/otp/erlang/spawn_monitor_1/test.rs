mod with_function;

use std::convert::TryInto;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::Status;
use liblumen_alloc::erts::term::{Pid, Term};

use crate::otp::erlang::spawn_monitor_1::native;
use crate::registry::pid_to_process;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_function_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_function(arc_process.clone()),
                |function| {
                    prop_assert_eq!(native(&arc_process, function), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}
