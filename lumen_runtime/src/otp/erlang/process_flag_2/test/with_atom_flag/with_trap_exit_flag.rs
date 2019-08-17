use super::*;

use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::term::{atom_unchecked, Term};

use crate::otp::erlang;
use crate::process;
use crate::scheduler::Scheduler;
use crate::test::has_message;

#[test]
fn without_boolean_value_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_boolean(arc_process.clone()),
                |value| {
                    prop_assert_eq!(native(&arc_process, flag(), value), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_boolean_returns_original_value_false() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(&strategy::term::is_boolean(), |value| {
            let arc_process = process::test(&process::test_init());

            prop_assert_eq!(native(&arc_process, flag(), value), Ok(false.into()));

            Ok(())
        })
        .unwrap();
}

#[test]
fn with_true_value_then_boolean_value_returns_old_value_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(&strategy::term::is_boolean(), |value| {
            let arc_process = process::test(&process::test_init());

            let old_value = true.into();
            prop_assert_eq!(native(&arc_process, flag(), old_value), Ok(false.into()));

            prop_assert_eq!(native(&arc_process, flag(), value), Ok(old_value));

            Ok(())
        })
        .unwrap();
}

#[test]
fn with_true_value_with_linked_receive_exit_message_and_does_not_exit_when_linked_process_exits() {
    with_process(|process| {
        let other_arc_process = process::test(process);

        process.link(&other_arc_process);

        assert_eq!(native(process, flag(), true.into()), Ok(false.into()));

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(!other_arc_process.is_exiting());
        assert!(!process.is_exiting());

        let reason = atom_unchecked("exit_reason");

        erlang::exit_1::place_frame_with_arguments(&other_arc_process, Placement::Replace, reason)
            .unwrap();

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(other_arc_process.is_exiting());
        assert!(!process.is_exiting());

        let tag = atom_unchecked("EXIT");
        let from = other_arc_process.pid_term();
        let exit_message = process.tuple_from_slice(&[tag, from, reason]).unwrap();

        assert!(has_message(process, exit_message));
    });
}

#[test]
fn with_true_value_then_false_value_exits_when_linked_process_exits() {
    with_process(|process| {
        let other_arc_process = process::test(process);

        process.link(&other_arc_process);

        assert_eq!(native(process, flag(), true.into()), Ok(false.into()));

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(!other_arc_process.is_exiting());
        assert!(!process.is_exiting());

        assert_eq!(native(process, flag(), false.into()), Ok(true.into()));

        let reason = atom_unchecked("exit_reason");

        erlang::exit_1::place_frame_with_arguments(&other_arc_process, Placement::Replace, reason)
            .unwrap();

        assert!(Scheduler::current().run_through(&other_arc_process));

        assert!(other_arc_process.is_exiting());
        assert!(process.is_exiting());
    });
}

fn flag() -> Term {
    atom_unchecked("trap_exit")
}
