mod with_arity_zero;

use super::*;

use proptest::strategy::Strategy;

use liblumen_alloc::erts::exception::Exception;

#[test]
fn without_arity_zero_returns_pid_to_parent_and_child_process_exits_badarity() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::module_function_arity::module(),
                    strategy::module_function_arity::function(),
                    (1_u8..=255_u8),
                )
                    .prop_map(|(module, function, arity)| {
                        strategy::term::export_closure(&arc_process, module, function, arity)
                    }),
                |function| {
                    let result = native(&arc_process, function, OPTIONS);

                    prop_assert!(result.is_ok());

                    let child_pid_term = result.unwrap();

                    prop_assert!(child_pid_term.is_pid());

                    let child_pid: Pid = child_pid_term.try_into().unwrap();

                    let child_arc_process = pid_to_process(&child_pid).unwrap();

                    let scheduler = Scheduler::current();

                    prop_assert!(scheduler.run_once());
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

                    Ok(())
                },
            )
            .unwrap();
    });
}

const OPTIONS: Term = Term::NIL;
