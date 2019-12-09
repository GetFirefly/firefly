mod with_pid;

use anyhow::*;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::Pid;

use crate::otp::erlang::is_process_alive_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_pid_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_pid(arc_process.clone()), |pid| {
                prop_assert_badarg!(
                    native(&arc_process, pid),
                    format!("pid ({}) must be a pid", pid)
                );

                Ok(())
            })
            .unwrap();
    });
}
