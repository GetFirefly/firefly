mod with_pid;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarg;

use crate::otp::erlang::is_process_alive_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_pid_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_pid(arc_process.clone()), |term| {
                prop_assert_eq!(native(&arc_process, term), Err(badarg!().into()));

                Ok(())
            })
            .unwrap();
    });
}
