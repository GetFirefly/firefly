use super::*;

#[path = "with_exported_function/w_a.rs"]
mod with_arity;

#[test]
fn without_arity_when_run_exits_undef_and_parent_does_not_exit() {
    let parent_arc_process = process::test_init();
    let arc_scheduler = Scheduler::current();

    let priority = Priority::Normal;
    let run_queue_length_before = arc_scheduler.run_queue_len(priority);

    let module_atom = Atom::try_from_str("erlang").unwrap();
    let module = unsafe { module_atom.as_term() };

    let function_atom = Atom::try_from_str("+").unwrap();
    let function = unsafe { function_atom.as_term() };

    // erlang.+/1 and erlang.+/2 exists so use 3 for invalid arity
    let arguments = parent_arc_process
        .list_from_slice(&[
            parent_arc_process.integer(0).unwrap(),
            parent_arc_process.integer(1).unwrap(),
            parent_arc_process.integer(2).unwrap(),
        ])
        .unwrap();

    let result = native(&parent_arc_process, module, function, arguments, OPTIONS);

    assert!(result.is_ok());

    let child_pid = result.unwrap();
    let child_pid_result_pid: core::result::Result<Pid, _> = child_pid.try_into();

    assert!(child_pid_result_pid.is_ok());

    let child_pid_pid = child_pid_result_pid.unwrap();

    let run_queue_length_after = arc_scheduler.run_queue_len(priority);

    assert_eq!(run_queue_length_after, run_queue_length_before + 1);

    let child_arc_process = pid_to_process(&child_pid_pid).unwrap();

    assert!(arc_scheduler.run_through(&child_arc_process));
    assert!(!arc_scheduler.run_through(&child_arc_process));

    assert_eq!(child_arc_process.code_stack_len(), 1);
    assert_eq!(
        child_arc_process.current_module_function_arity(),
        Some(apply_3::module_function_arity())
    );

    match *child_arc_process.status.read() {
        Status::Exiting(ref runtime_exception) => {
            let runtime_undef: runtime::Exception =
                undef!(&child_arc_process, module, function, arguments)
                    .try_into()
                    .unwrap();

            assert_eq!(runtime_exception, &runtime_undef);
        }
        ref status => panic!("Process status ({:?}) is not exiting.", status),
    };

    assert!(!parent_arc_process.is_exiting());
}
