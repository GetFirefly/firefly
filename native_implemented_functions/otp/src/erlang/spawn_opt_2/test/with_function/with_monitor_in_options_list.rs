mod with_arity_zero;

use super::*;

#[test]
fn without_arity_zero_returns_pid_to_parent_and_child_process_exits_badarity_and_sends_exit_message_to_parent(
) {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(
                strategy::module_function_arity::module(),
                strategy::module_function_arity::function(),
                strategy::term::export_closure_non_zero_arity_range_inclusive(),
            ),
            |(module, function, arity)| {
                let parent_arc_process = test::process::init();
                let function =
                    strategy::term::export_closure(&parent_arc_process, module, function, arity);

                let result = result(&parent_arc_process, function, options(&parent_arc_process));

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

                let args = Term::NIL;

                prop_assert_exits_badarity(
                    &child_arc_process,
                    function,
                    args,
                    format!(
                        "arguments ([]) length (0) does not match arity ({}) of function ({})",
                        arity, function
                    ),
                )?;

                prop_assert!(!parent_arc_process.is_exiting());

                let tag = atom!("DOWN");
                let reason = badarity_reason(&parent_arc_process, function, args);

                prop_assert!(has_message(
                    &parent_arc_process,
                    parent_arc_process
                        .tuple_from_slice(&[
                            tag,
                            monitor_reference,
                            atom!("process"),
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

fn option() -> Term {
    Atom::str_to_term("monitor")
}

fn options(process: &Process) -> Term {
    process.list_from_slice(&[option()]).unwrap()
}
