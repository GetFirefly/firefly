mod spawn_apply_3;

use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::term::prelude::Atom;

use crate::otp::erlang::exit_1;
use crate::scheduler::{with_process_arc, Scheduler};

#[test]
fn scheduler_does_not_requeue_exiting_process() {
    with_process_arc(|arc_process| {
        exit_1::place_frame_with_arguments(
            &arc_process,
            Placement::Replace,
            Atom::str_to_term("normal"),
        )
        .unwrap();

        let scheduler = Scheduler::current();

        assert!(scheduler.is_run_queued(&arc_process));

        assert!(scheduler.run_through(&arc_process));

        assert!(!scheduler.is_run_queued(&arc_process));
    })
}

#[test]
fn scheduler_does_run_exiting_process() {
    with_process_arc(|arc_process| {
        let scheduler = Scheduler::current();

        assert!(scheduler.is_run_queued(&arc_process));
        assert!(scheduler.run_through(&arc_process));
        assert!(scheduler.is_run_queued(&arc_process));

        arc_process.exit_normal();

        assert!(scheduler.is_run_queued(&arc_process));
        assert!(!scheduler.run_through(&arc_process));
        assert!(!scheduler.is_run_queued(&arc_process));
    })
}
