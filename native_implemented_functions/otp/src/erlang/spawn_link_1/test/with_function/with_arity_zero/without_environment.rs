use super::*;

#[test]
fn without_expected_exit_in_child_process_exits_linked_parent_process() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(
                strategy::module_function_arity::module(),
                strategy::module_function_arity::function(),
            )
                .prop_map(|(module, function)| {
                    let arc_process = test::process::init();
                    let arity = 0;
                    let code = |arc_process: &Arc<Process>| {
                        arc_process.exception(exit!(
                            Atom::str_to_term("not_normal"),
                            anyhow!("Test").into()
                        ));

                        Ok(())
                    };

                    (
                        arc_process.clone(),
                        arc_process
                            .export_closure(module, function, arity, Some(code))
                            .unwrap(),
                    )
                }),
            |(parent_arc_process, function)| {
                let result = native(&parent_arc_process, function);

                prop_assert!(result.is_ok());

                let child_pid_term = result.unwrap();

                prop_assert!(child_pid_term.is_pid());

                let child_pid: Pid = child_pid_term.try_into().unwrap();

                let child_arc_process = pid_to_process(&child_pid).unwrap();

                let scheduler = scheduler::current();

                prop_assert!(scheduler.run_once());
                prop_assert!(scheduler.run_once());

                match *child_arc_process.status.read() {
                    Status::Exiting(ref exception) => {
                        prop_assert_eq!(
                            exception,
                            &exit!(Atom::str_to_term("not_normal"), anyhow!("Test").into())
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
                        prop_assert_eq!(
                            exception,
                            &exit!(Atom::str_to_term("not_normal"), anyhow!("Test").into())
                        );
                    }
                    ref status => {
                        return Err(proptest::test_runner::TestCaseError::fail(format!(
                            "Parent process did not exit.  Status is {:?}",
                            status
                        )))
                    }
                }

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_expected_exit_in_child_process_does_not_exit_linked_parent_process() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(
                strategy::module_function_arity::module(),
                strategy::module_function_arity::function(),
            )
                .prop_map(|(module, function)| {
                    let arc_process = test::process::init();
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
                }),
            |(parent_arc_process, function)| {
                let result = native(&parent_arc_process, function);

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

                match *parent_arc_process.status.read() {
                    Status::Exiting(ref exception) => {
                        return Err(proptest::test_runner::TestCaseError::fail(format!(
                            "Parent process exited {:?}",
                            exception
                        )))
                    }
                    _ => (),
                }

                Ok(())
            },
        )
        .unwrap();
}
