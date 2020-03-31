use super::*;

#[test]
fn without_expected_exit_in_child_process_sends_exit_message_to_parent() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(
                strategy::module_function_arity::module(),
                function::anonymous::index(),
                function::anonymous::old_unique(),
                function::anonymous::unique(),
            )
                .prop_map(|(module, index, old_unique, unique)| {
                    let arc_process = test::process::init();
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
                }),
            |(parent_arc_process, function)| {
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

                let scheduler = scheduler::current();

                prop_assert!(scheduler.run_once());
                prop_assert!(scheduler.run_once());

                let reason = child_arc_process
                    .list_from_slice(&[Atom::str_to_term("first"), Atom::str_to_term("second")])
                    .unwrap();

                match *child_arc_process.status.read() {
                    Status::Exiting(ref exception) => {
                        prop_assert_eq!(exception, &exit!(reason, anyhow!("Test").into()));
                    }
                    ref status => {
                        return Err(proptest::test_runner::TestCaseError::fail(format!(
                            "Child process did not exit.  Status is {:?}",
                            status
                        )))
                    }
                }

                prop_assert!(!parent_arc_process.is_exiting());

                let tag = Atom::str_to_term("DOWN");

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

#[test]
fn with_expected_exit_in_child_process_send_exit_message_to_parent() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(
                strategy::module_function_arity::module(),
                function::anonymous::index(),
                function::anonymous::old_unique(),
                function::anonymous::unique(),
            )
                .prop_map(|(module, index, old_unique, unique)| {
                    let arc_process = test::process::init();
                    let creator = arc_process.pid().into();
                    let arity = 0;
                    let code = |arc_process: &Arc<Process>| {
                        let first = arc_process.stack_pop().unwrap();
                        let second = arc_process.stack_pop().unwrap();
                        let reason = arc_process.tuple_from_slice(&[first, second])?;

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
                                &[
                                    Atom::str_to_term("shutdown"),
                                    Atom::str_to_term("shutdown_reason"),
                                ],
                            )
                            .unwrap(),
                    )
                }),
            |(parent_arc_process, function)| {
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

                let scheduler = scheduler::current();

                prop_assert!(scheduler.run_once());
                prop_assert!(scheduler.run_once());

                let reason = child_arc_process
                    .tuple_from_slice(&[
                        Atom::str_to_term("shutdown"),
                        Atom::str_to_term("shutdown_reason"),
                    ])
                    .unwrap();

                match *child_arc_process.status.read() {
                    Status::Exiting(ref exception) => {
                        prop_assert_eq!(exception, &exit!(reason, anyhow!("Test").into()));
                    }
                    ref status => {
                        return Err(proptest::test_runner::TestCaseError::fail(format!(
                            "Child process did not exit.  Status is {:?}",
                            status
                        )))
                    }
                }

                prop_assert!(!parent_arc_process.is_exiting());

                let tag = Atom::str_to_term("DOWN");

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
