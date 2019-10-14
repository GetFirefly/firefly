use super::*;

use std::sync::Arc;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::atom_unchecked;
use liblumen_alloc::erts::ModuleFunctionArity;
use liblumen_alloc::exit;

#[test]
fn without_expected_exit_in_child_process_exits_linked_parent_process() {
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
                        arc_process.exception(exit!(atom_unchecked("not_normal")));

                        Ok(())
                    };

                    (
                        arc_process.clone(),
                        arc_process
                            .closure_with_env_from_slice(module_function_arity, code, creator, &[])
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

                let scheduler = Scheduler::current();

                prop_assert!(scheduler.run_once());
                prop_assert!(scheduler.run_once());

                match *child_arc_process.status.read() {
                    Status::Exiting(ref exception) => {
                        prop_assert_eq!(exception, &exit!(atom_unchecked("not_normal")));
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
                        prop_assert_eq!(exception, &exit!(atom_unchecked("normal")));
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
                    let arc_process = process::test_init();
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

                    (
                        arc_process.clone(),
                        arc_process
                            .closure_with_env_from_slice(module_function_arity, code, creator, &[])
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

                let scheduler = Scheduler::current();

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
