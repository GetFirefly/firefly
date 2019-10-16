mod with_arity_zero;

use super::*;

use proptest::prop_assert;
use proptest::strategy::Strategy;

use liblumen_alloc::badarity;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::{atom_unchecked, Boxed, Tuple};

use crate::process;
use crate::scheduler::Scheduler;
use crate::test::has_message;

#[test]
fn without_arity_zero_returns_pid_to_parent_and_child_process_exits_badarity_and_sends_exit_message_to_parent(
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
                    strategy::term::closure(&parent_arc_process, module, function, arity);

                let result = native(&parent_arc_process, function);

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
                            Exception::Runtime(*exception),
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

                prop_assert!(!parent_arc_process.is_exiting());

                let tag = atom_unchecked("DOWN");
                let reason = match badarity!(&parent_arc_process, function, Term::NIL) {
                    Exception::Runtime(runtime_exception) => runtime_exception.reason,
                    _ => unreachable!("parent process out-of-memory"),
                };

                prop_assert!(has_message(
                    &parent_arc_process,
                    parent_arc_process
                        .tuple_from_slice(&[
                            tag,
                            monitor_reference,
                            atom_unchecked("process"),
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
