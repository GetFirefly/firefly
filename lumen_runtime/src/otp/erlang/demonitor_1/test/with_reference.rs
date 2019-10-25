mod with_monitor;

use super::*;

use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::term::prelude::Atom;

use crate::otp::erlang::exit_1;
use crate::otp::erlang::monitor_2;
use crate::process;
use crate::process::SchedulerDependentAlloc;
use crate::scheduler::Scheduler;
use crate::test::{has_message, monitor_count, monitored_count};

#[test]
fn without_monitor_returns_true() {
    with_process_arc(|monitoring_arc_process| {
        let reference = monitoring_arc_process.next_reference().unwrap();

        assert_eq!(native(&monitoring_arc_process, reference), Ok(true.into()))
    });
}
