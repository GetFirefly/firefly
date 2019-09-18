mod with_local_reference;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{atom_unchecked, AsTerm, Term};

use crate::otp::erlang;
use crate::otp::erlang::read_timer_1::native;
use crate::process;
use crate::process::SchedulerDependentAlloc;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;
use crate::test::{has_message, receive_message, timeout_message};
use crate::time::monotonic::Milliseconds;
use crate::timer;

#[test]
fn without_reference_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_reference(arc_process.clone()),
                |timer_reference| {
                    prop_assert_eq!(native(&arc_process, timer_reference), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}
