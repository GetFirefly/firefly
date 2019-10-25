use super::*;

use std::sync::Arc;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::atom_unchecked;
use liblumen_alloc::erts::ModuleFunctionArity;
use liblumen_alloc::exit;

#[test]
fn without_environment_runs_function_in_child_process() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::module_function_arity::module(),
                    strategy::module_function_arity::function(),
                )
                    .prop_map(|(module, function)| {
                        let creator = arc_process.pid_term();
                        let module_function_arity = Arc::new(ModuleFunctionArity {
                            module,
                            function,
                            arity: 0,
                        });
                        let code = |arc_process: &Arc<Process>| {
                            arc_process.return_from_call(atom_unchecked("ok"))?;

                            Ok(())
                        };

                        arc_process
                            .closure_with_env_from_slice(module_function_arity, code, creator, &[])
                            .unwrap()
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
                            prop_assert_eq!(exception, &exit!(atom_unchecked("normal")));
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

#[test]
fn with_environment_runs_function_in_child_process() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::module_function_arity::module(),
                    strategy::module_function_arity::function(),
                )
                    .prop_map(|(module, function)| {
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

                        arc_process
                            .closure_with_env_from_slice(
                                module_function_arity,
                                code,
                                creator,
                                &[atom_unchecked("first"), atom_unchecked("second")],
                            )
                            .unwrap()
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
                                exception,
                                &exit!(child_arc_process
                                    .list_from_slice(&[
                                        atom_unchecked("first"),
                                        atom_unchecked("second")
                                    ])
                                    .unwrap())
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
