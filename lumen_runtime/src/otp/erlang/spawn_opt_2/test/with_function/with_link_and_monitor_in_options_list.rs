mod with_arity_zero;

use super::*;

use proptest::prop_assert;
use proptest::strategy::Strategy;

use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{badarity, exit};

use crate::process;
use crate::scheduler::Scheduler;
use crate::test::has_message;

#[test]
fn without_arity_zero_returns_pid_to_parent_and_child_process_exits_badarity_and_sends_exit_message_to_and_exits_parent(
) {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(
                strategy::module_function_arity::module(),
                strategy::module_function_arity::function(),
                (1_u8..=255_u8),
            ),
            |(module, function, arity)| {
                let parent_arc_process = process::test_init();
                let function =
                    strategy::term::export_closure(&parent_arc_process, module, function, arity);

                let result = native(&parent_arc_process, function, options(&parent_arc_process));

                prop_assert!(result.is_ok());

                let result_boxed_tuple: Result<Boxed<Tuple>, _> = result.unwrap().try_into();

                prop_assert!(result_boxed_tuple.is_ok());

                let boxed_tuple = result_boxed_tuple.unwrap();

                prop_assert_eq!(boxed_tuple.len(), 2);

                let child_pid_term = boxed_tuple[0];

                prop_assert!(child_pid_term.is_pid());

                let child_pid: Pid = child_pid_term.try_into().unwrap();
                let child_arc_process = pid_to_process(&child_pid).unwrap();

                let monitor_reference = boxed_tuple[1];

                prop_assert!(monitor_reference.is_reference());

                let scheduler = Scheduler::current();

                prop_assert!(scheduler.run_once());
                prop_assert!(scheduler.run_once());

                match *child_arc_process.status.read() {
                    Status::Exiting(ref exception) => {
                        prop_assert_eq!(
                            Exception::Runtime(exception.clone()),
                            badarity!(&child_arc_process, function, Term::NIL)
                        );
                    }
                    ref status => {
                        return Err(proptest::test_runner::TestCaseError::fail(format!(
                            "Child process did not exit.  Status is {:?}",
                            status
                        )))
                    }
                }

                match *parent_arc_process.status.read() {
                    Status::Exiting(ref exception) => {
                        let reason = match badarity!(&parent_arc_process, function, Term::NIL) {
                            Exception::Runtime(badarity_runtime_exception) => {
                                (badarity_runtime_exception.reason().unwrap())
                            }
                            _ => unreachable!("parent process out-of-memory"),
                        };

                        prop_assert_eq!(exception, &exit!(&parent_arc_process, reason));
                    }
                    ref status => {
                        return Err(proptest::test_runner::TestCaseError::fail(format!(
                            "Parent process did not exit.  Status is {:?}",
                            status
                        )))
                    }
                }

                let tag = Atom::str_to_term("DOWN");
                let reason = match badarity!(&parent_arc_process, function, Term::NIL) {
                    Exception::Runtime(runtime_exception) => runtime_exception.reason().unwrap(),
                    _ => unreachable!("parent process out-of-memory"),
                };

                prop_assert!(has_message(
                    &parent_arc_process,
                    parent_arc_process
                        .tuple_from_slice(&[
                            tag,
                            monitor_reference,
                            Atom::str_to_term("process"),
                            child_pid_term,
                            reason
                        ])
                        .unwrap()
                ));

                Ok(())
            },
        )
        .unwrap();
}

fn options(process: &Process) -> Term {
    process
        .list_from_slice(&[Atom::str_to_term("link"), Atom::str_to_term("monitor")])
        .unwrap()
}
