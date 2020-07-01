use super::*;

use anyhow::*;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Atom;
use liblumen_alloc::exit;

#[test]
fn without_environment_runs_function_in_child_process() {
    let arc_process = process::default();
    let function = anonymous_0::anonymous_closure(&arc_process).unwrap();
    let result = result(&arc_process, function);

    assert!(result.is_ok());

    let child_pid_term = result.unwrap();

    assert!(child_pid_term.is_pid());

    let child_pid: Pid = child_pid_term.try_into().unwrap();

    let child_arc_process = pid_to_process(&child_pid).unwrap();

    assert!(scheduler::run_through(&child_arc_process));

    match *child_arc_process.status.read() {
        Status::RuntimeException(ref exception) => {
            assert_eq!(
                exception,
                &exit!(Atom::str_to_term("normal"), anyhow!("Test").into())
            );
        }
        ref status => panic!("Child process did not exit.  Status is {:?}", status),
    };
}

#[test]
fn with_environment_runs_function_in_child_process() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::module_function_arity::module(),
                function::anonymous::index(),
                function::anonymous::old_unique(),
                function::anonymous::unique(),
            )
                .prop_map(|(arc_process, module, index, old_unique, unique)| {
                    let arity = 0;
                    let creator = arc_process.pid().into();

                    fn result(
                        process: &Process,
                        first_environment: Term,
                        second_environment: Term,
                    ) -> exception::Result<Term> {
                        let reason =
                            process.list_from_slice(&[first_environment, second_environment])?;

                        Err(exit!(reason, anyhow!("Test").into()).into())
                    }

                    extern "C" fn native(
                        first_environment: Term,
                        second_environment: Term,
                    ) -> Term {
                        let arc_process = current_process();
                        arc_process.reduce();

                        arc_process.return_status(result(
                            &arc_process,
                            first_environment,
                            second_environment,
                        ))
                    }

                    (
                        arc_process.clone(),
                        arc_process
                            .anonymous_closure_with_env_from_slice(
                                module,
                                index,
                                old_unique,
                                unique,
                                arity,
                                Some(native as _),
                                creator,
                                &[Atom::str_to_term("first"), Atom::str_to_term("second")],
                            )
                            .unwrap(),
                    )
                })
        },
        |(arc_process, function)| {
            let result = result(&arc_process, function);

            prop_assert!(result.is_ok());

            let child_pid_term = result.unwrap();

            prop_assert!(child_pid_term.is_pid());

            let child_pid: Pid = child_pid_term.try_into().unwrap();
            let child_arc_process = pid_to_process(&child_pid).unwrap();

            prop_assert!(scheduler::run_through(&child_arc_process));

            match *child_arc_process.status.read() {
                Status::RuntimeException(ref exception) => {
                    prop_assert_eq!(
                        exception,
                        &exit!(
                            child_arc_process
                                .list_from_slice(&[
                                    Atom::str_to_term("first"),
                                    Atom::str_to_term("second")
                                ])
                                .unwrap(),
                            anyhow!("Test").into()
                        )
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
    );
}
