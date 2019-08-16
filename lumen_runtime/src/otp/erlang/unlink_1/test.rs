mod with_local_pid;

use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::ProcessControlBlock;

use crate::otp::erlang::unlink_1::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;

#[test]
fn without_pid_or_port_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term(arc_process.clone())
                    .prop_filter("Cannot be pid or port", |pid_or_port| {
                        !(pid_or_port.is_pid() || pid_or_port.is_port())
                    }),
                |pid_or_port| {
                    prop_assert_eq!(native(&arc_process, pid_or_port), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn link_count(process: &ProcessControlBlock) -> usize {
    process.linked_pid_set.lock().len()
}
