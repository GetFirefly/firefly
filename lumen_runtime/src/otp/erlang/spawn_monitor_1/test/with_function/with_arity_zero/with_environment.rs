use super::*;

use std::sync::Arc;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::atom_unchecked;
use liblumen_alloc::erts::ModuleFunctionArity;
use liblumen_alloc::exit;

#[test]
fn without_expected_exit_in_child_process_sends_exit_message_to_parent() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(
                strategy::module_function_arity::module(),
                strategy::module_function_arity::function(),
            )
                .prop_map(|(module, function)| {
                    let arc_process = process::test_init();
                    let creator = arc_process.pid_term();
                    let module_function_arity = Arc::new(ModuleFunctionArity {
                        module,
                        function,
                        arity: 0,
                    });
                    let code = |arc_process: &Arc<Process>| {
                        let first = arc_process.stack_pop().unwrap();
                        let second = arc_process.stack_pop().unwrap();
                        let reason = arc_process.list_from_slice(&[first, second])?;

                        arc_process.exception(exit!(reason));

                        Ok(())
                    };

                    (
                        arc_process.clone(),
                        arc_process
                            .closure_with_env_from_slice(
                                module_function_arity,
                                code,
                                creator,
                                &[atom_unchecked("first"), atom_unchecked("second")],
                            )
                            .unwrap(),
                    )
                }),
            |(parent_arc_process, function)| {
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

                let reason = child_arc_process
                    .list_from_slice(&[atom_unchecked("first"), atom_unchecked("second")])
                    .unwrap();

                match *child_arc_process.status.read() {
                    Status::Exiting(ref exception) => {
                        prop_assert_eq!(exception, &exit!(reason));
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

#[test]
fn with_expected_exit_in_child_process_send_exit_message_to_parent() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(
                strategy::module_function_arity::module(),
                strategy::module_function_arity::function(),
            )
                .prop_map(|(module, function)| {
                    let arc_process = process::test_init();
                    let creator = arc_process.pid_term();
                    let module_function_arity = Arc::new(ModuleFunctionArity {
                        module,
                        function,
                        arity: 0,
                    });
                    let code = |arc_process: &Arc<Process>| {
                        let first = arc_process.stack_pop().unwrap();
                        let second = arc_process.stack_pop().unwrap();
                        let reason = arc_process.tuple_from_slice(&[first, second])?;

                        arc_process.exception(exit!(reason));

                        Ok(())
                    };

                    (
                        arc_process.clone(),
                        arc_process
                            .closure_with_env_from_slice(
                                module_function_arity,
                                code,
                                creator,
                                &[
                                    atom_unchecked("shutdown"),
                                    atom_unchecked("shutdown_reason"),
                                ],
                            )
                            .unwrap(),
                    )
                }),
            |(parent_arc_process, function)| {
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

                let reason = child_arc_process
                    .tuple_from_slice(&[
                        atom_unchecked("shutdown"),
                        atom_unchecked("shutdown_reason"),
                    ])
                    .unwrap();

                match *child_arc_process.status.read() {
                    Status::Exiting(ref exception) => {
                        prop_assert_eq!(exception, &exit!(reason));
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
