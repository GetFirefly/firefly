use super::*;

use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::term::prelude::{Atom, Pid};
use liblumen_alloc::{atom, error};

use crate::otp::erlang;
use crate::process;
use crate::scheduler::Scheduler;

#[test]
fn with_self_returns_true_but_does_not_create_link() {
    with_process(|process| {
        let link_count_before = link_count(process);

        assert_eq!(native(process, process.pid_term()), Ok(true.into()));

        assert_eq!(link_count(process), link_count_before);
    });
}

#[test]
fn with_non_existent_pid_errors_noproc() {
    with_process(|process| {
        let link_count_before = link_count(process);

        assert_eq!(
            native(process, Pid::next_term()),
            Err(error!(process, atom!("noproc")).into())
        );

        assert_eq!(link_count(process), link_count_before);
    });
}

#[test]
fn with_existing_unlinked_pid_links_to_process() {
    with_process(|process| {
        let other_process = process::test(process);

        let process_link_count_before = link_count(process);
        let other_process_link_count_before = link_count(process);

        assert_eq!(native(process, other_process.pid_term()), Ok(true.into()));

        assert_eq!(link_count(process), process_link_count_before + 1);
        assert_eq!(
            link_count(&other_process),
            other_process_link_count_before + 1
        );
    });
}

#[test]
fn with_existing_linked_pid_returns_true() {
    with_process(|process| {
        let other_process = process::test(process);

        process.link(&other_process);

        let process_link_count_before = link_count(process);
        let other_process_link_count_before = link_count(process);

        assert_eq!(native(process, other_process.pid_term()), Ok(true.into()));

        assert_eq!(link_count(process), process_link_count_before);
        assert_eq!(link_count(&other_process), other_process_link_count_before);
    });
}

#[test]
fn when_a_linked_process_exits_normal_the_process_does_not_exit() {
    with_process(|process| {
        let other_arc_process = process::test(process);

        assert_eq!(
            native(process, other_arc_process.pid_term()),
            Ok(true.into())
        );

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(!other_arc_process.is_exiting());
        assert!(!process.is_exiting());

        erlang::exit_1::place_frame_with_arguments(
            &other_arc_process,
            Placement::Replace,
            Atom::str_to_term("normal"),
        )
        .unwrap();

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(other_arc_process.is_exiting());
        assert!(!process.is_exiting())
    });
}

#[test]
fn when_a_linked_process_exits_shutdown_the_process_does_not_exit() {
    with_process(|process| {
        let other_arc_process = process::test(process);

        assert_eq!(
            native(process, other_arc_process.pid_term()),
            Ok(true.into())
        );

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(!other_arc_process.is_exiting());
        assert!(!process.is_exiting());

        erlang::exit_1::place_frame_with_arguments(
            &other_arc_process,
            Placement::Replace,
            Atom::str_to_term("shutdown"),
        )
        .unwrap();

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(other_arc_process.is_exiting());
        assert!(!process.is_exiting())
    });
}

#[test]
fn when_a_linked_process_exits_with_shutdown_tuple_the_process_does_not_exit() {
    with_process(|process| {
        let other_arc_process = process::test(process);

        assert_eq!(
            native(process, other_arc_process.pid_term()),
            Ok(true.into())
        );

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(!other_arc_process.is_exiting());
        assert!(!process.is_exiting());

        let tag = Atom::str_to_term("shutdown");
        let shutdown_reason = Atom::str_to_term("test");
        let reason = other_arc_process
            .tuple_from_slice(&[tag, shutdown_reason])
            .unwrap();
        erlang::exit_1::place_frame_with_arguments(&other_arc_process, Placement::Replace, reason)
            .unwrap();

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(other_arc_process.is_exiting());
        assert!(!process.is_exiting())
    });
}

#[test]
fn when_a_linked_process_exits_unexpected_the_process_does_not_exit() {
    with_process(|process| {
        let other_arc_process = process::test(process);

        assert_eq!(
            native(process, other_arc_process.pid_term()),
            Ok(true.into())
        );

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(!other_arc_process.is_exiting());
        assert!(!process.is_exiting());

        erlang::exit_1::place_frame_with_arguments(
            &other_arc_process,
            Placement::Replace,
            Atom::str_to_term("abnormal"),
        )
        .unwrap();

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(other_arc_process.is_exiting());
        assert!(process.is_exiting())
    });
}

#[test]
fn when_the_process_exits_unexpected_linked_processes_exit_too() {
    with_process_arc(|arc_process| {
        let other_arc_process = process::test(&arc_process);

        assert_eq!(
            native(&arc_process, other_arc_process.pid_term()),
            Ok(true.into())
        );

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(!other_arc_process.is_exiting());
        assert!(!arc_process.is_exiting());

        erlang::exit_1::place_frame_with_arguments(
            &arc_process,
            Placement::Replace,
            Atom::str_to_term("abnormal"),
        )
        .unwrap();

        assert!(Scheduler::current().run_through(&arc_process));

        assert!(other_arc_process.is_exiting());
        assert!(arc_process.is_exiting())
    });
}
