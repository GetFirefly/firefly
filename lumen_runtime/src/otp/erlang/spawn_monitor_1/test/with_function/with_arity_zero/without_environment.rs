use super::*;

use std::sync::Arc;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Atom;
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
                    let arity = 0;
                    let code = |arc_process: &Arc<Process>| {
                        arc_process.exception(exit!(arc_process, atom!("not_normal")));

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

                let returned = result.unwrap();

                let result_boxed_tuple: Result<Boxed<Tuple>, _> = returned.try_into();

                prop_assert!(
                    result_boxed_tuple.is_ok(),
                    "Returned ({:?}) is not a tuple",
                    returned
                );

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

                let reason = Atom::str_to_term("not_normal");

                match *child_arc_process.status.read() {
                    Status::Exiting(ref exception) => {
                        prop_assert_eq!(exception, &exit!(&child_arc_process, reason));
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
fn with_expected_exit_in_child_process_sends_exit_message_to_parent() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(
                strategy::module_function_arity::module(),
                strategy::module_function_arity::function(),
            )
                .prop_map(|(module, function)| {
                    let arc_process = process::test_init();
                    let arity = 0;
                    let code = |arc_process: &Arc<Process>| {
                        arc_process.return_from_call(Atom::str_to_term("ok"))?;

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

                let reason = Atom::str_to_term("normal");

                match *child_arc_process.status.read() {
                    Status::Exiting(ref exception) => {
                        prop_assert_eq!(exception, &exit!(&child_arc_process, reason));
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
