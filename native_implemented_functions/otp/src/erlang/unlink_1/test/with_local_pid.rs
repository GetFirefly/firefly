use super::*;

use liblumen_alloc::erts::term::prelude::{Atom, Pid};

use crate::runtime::scheduler;

use crate::test;
use crate::test::exit_when_run;

#[test]
fn with_self_returns_true() {
    with_process(|process| {
        let link_count_before = link_count(process);

        assert_eq!(result(process, process.pid_term()), Ok(true.into()));

        assert_eq!(link_count(process), link_count_before);
    });
}

#[test]
fn with_non_existent_pid_returns_true() {
    with_process(|process| {
        let link_count_before = link_count(process);

        assert_eq!(result(process, Pid::next_term()), Ok(true.into()));

        assert_eq!(link_count(process), link_count_before);
    });
}

#[test]
fn with_existing_unlinked_pid_returns_true() {
    with_process(|process| {
        let other_process = test::process::child(process);

        let process_link_count_before = link_count(process);
        let other_process_link_count_before = link_count(process);

        assert_eq!(result(process, other_process.pid_term()), Ok(true.into()));

        assert_eq!(link_count(process), process_link_count_before);
        assert_eq!(link_count(&other_process), other_process_link_count_before);
    });
}

#[test]
fn with_existing_linked_pid_unlinks_processes_and_returns_true() {
    with_process(|process| {
        let other_process = test::process::child(process);

        process.link(&other_process);

        let process_link_count_before = link_count(process);
        let other_process_link_count_before = link_count(process);

        assert_eq!(result(process, other_process.pid_term()), Ok(true.into()));

        assert_eq!(link_count(process), process_link_count_before - 1);
        assert_eq!(
            link_count(&other_process),
            other_process_link_count_before - 1
        );
    });
}

#[test]
fn when_a_linked_then_unlinked_process_exits_the_process_does_not_exit() {
    with_process(|process| {
        let other_arc_process = test::process::child(process);

        process.link(&other_arc_process);

        assert_eq!(
            result(process, other_arc_process.pid_term()),
            Ok(true.into())
        );

        assert!(scheduler::run_through(&other_arc_process));

        assert!(!other_arc_process.is_exiting());
        assert!(!process.is_exiting());

        exit_when_run(&other_arc_process, Atom::str_to_term("normal"));

        assert!(scheduler::run_through(&other_arc_process));

        assert!(other_arc_process.is_exiting());
        assert!(!process.is_exiting())
    });
}

#[test]
fn when_the_process_exits_the_linked_and_then_unlinked_process_exits_too() {
    with_process_arc(|arc_process| {
        let other_arc_process = test::process::child(&arc_process);

        arc_process.link(&other_arc_process);

        assert_eq!(
            result(&arc_process, other_arc_process.pid_term()),
            Ok(true.into())
        );

        assert!(scheduler::run_through(&other_arc_process));

        assert!(!other_arc_process.is_exiting());
        assert!(!arc_process.is_exiting());

        exit_when_run(&arc_process, Atom::str_to_term("normal"));

        assert!(scheduler::run_through(&arc_process));

        assert!(!other_arc_process.is_exiting());
        assert!(arc_process.is_exiting())
    });
}
