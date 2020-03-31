use super::*;

#[test]
fn without_environment_runs_function_in_child_process() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::module_function_arity::module(),
                strategy::module_function_arity::function(),
            )
                .prop_map(|(arc_process, module, function)| {
                    let arity = 0;
                    let code = |arc_process: &Arc<Process>| {
                        arc_process.return_from_call(0, Atom::str_to_term("ok"))?;

                        Ok(())
                    };

                    (
                        arc_process.clone(),
                        arc_process
                            .export_closure(module, function, arity, Some(code))
                            .unwrap(),
                    )
                })
        },
        |(arc_process, function)| {
            let result = native(&arc_process, function, options(&arc_process));

            prop_assert!(result.is_ok());

            let child_pid_term = result.unwrap();

            prop_assert!(child_pid_term.is_pid());

            let child_pid: Pid = child_pid_term.try_into().unwrap();

            let child_arc_process = pid_to_process(&child_pid).unwrap();

            prop_assert!(scheduler::run_through(&child_arc_process));
            prop_assert!(scheduler::run_through(&child_arc_process));

            match *child_arc_process.status.read() {
                Status::Exiting(ref exception) => {
                    prop_assert_eq!(
                        exception,
                        &exit!(Atom::str_to_term("normal"), anyhow!("Test").into())
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
                    let creator = arc_process.pid().into();
                    let arity = 0;
                    let code = |arc_process: &Arc<Process>| {
                        let first = arc_process.stack_pop().unwrap();
                        let second = arc_process.stack_pop().unwrap();
                        let reason = arc_process.list_from_slice(&[first, second])?;

                        arc_process.exception(exit!(reason, anyhow!("Test").into()));

                        Ok(())
                    };

                    (
                        arc_process.clone(),
                        arc_process
                            .anonymous_closure_with_env_from_slice(
                                module,
                                index,
                                old_unique,
                                unique,
                                arity,
                                Some(code),
                                creator,
                                &[Atom::str_to_term("first"), Atom::str_to_term("second")],
                            )
                            .unwrap(),
                    )
                })
        },
        |(arc_process, function)| {
            let result = native(&arc_process, function, options(&arc_process));

            prop_assert!(result.is_ok());

            let child_pid_term = result.unwrap();

            prop_assert!(child_pid_term.is_pid());

            let child_pid: Pid = child_pid_term.try_into().unwrap();

            let child_arc_process = pid_to_process(&child_pid).unwrap();

            prop_assert!(scheduler::run_through(&child_arc_process));

            match *child_arc_process.status.read() {
                Status::Exiting(ref exception) => {
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
