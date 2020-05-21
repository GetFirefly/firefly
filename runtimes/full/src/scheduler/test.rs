mod erlang;

use anyhow::*;

use liblumen_alloc::erts::term::prelude::*;

use crate::scheduler::{Scheduled, Scheduler};
use crate::{scheduler, test};

#[test]
fn scheduler_does_not_requeue_exiting_process() {
    let arc_process = test::process::default();

    let reason = Atom::str_to_term("normal");
    arc_process.queue_frame_with_arguments(erlang::exit_1::frame().with_arguments(false, &[reason]));
    arc_process.stack_queued_frames_with_arguments();
    arc_process.scheduler().unwrap().stop_waiting(&arc_process);

    let arc_dyn_scheduler = scheduler::current();
    let scheduler = arc_dyn_scheduler
        .as_any()
        .downcast_ref::<Scheduler>()
        .unwrap();

    assert!(scheduler.is_run_queued(&arc_process));
    assert!(scheduler::run_through(&arc_process));
    assert!(!scheduler.is_run_queued(&arc_process));
}

#[test]
fn scheduler_does_run_exiting_process() {
    let arc_process = test::process::default();
    let arc_dyn_scheduler = scheduler::current();
    let scheduler = arc_dyn_scheduler
        .as_any()
        .downcast_ref::<Scheduler>()
        .unwrap();

    assert!(scheduler.is_run_queued(&arc_process));
    assert!(scheduler::run_through(&arc_process));
    assert!(scheduler.is_run_queued(&arc_process));

    arc_process.exit_normal(anyhow!("Test").into());

    assert!(scheduler.is_run_queued(&arc_process));
    assert!(!scheduler::run_through(&arc_process));
    assert!(!scheduler.is_run_queued(&arc_process));
}

