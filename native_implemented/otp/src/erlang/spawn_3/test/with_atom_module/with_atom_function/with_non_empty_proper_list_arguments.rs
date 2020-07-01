use super::*;

mod with_loaded_module;

#[test]
fn without_loaded_module_when_run_exits_undef_and_parent_does_not_exit() {
    let parent_arc_process = test::process::init();
    let arc_scheduler = scheduler::current();

    let priority = Priority::Normal;
    let run_queue_length_before = arc_scheduler.run_queue_len(priority);

    // Typo
    let module = atom!("erlan");
    let function = atom!("+");

    let arguments = parent_arc_process
        .cons(parent_arc_process.integer(0).unwrap(), Term::NIL)
        .unwrap();

    let result = spawn_3::result(&parent_arc_process, module, function, arguments);

    assert!(result.is_ok());

    let child_pid = result.unwrap();
    let child_pid_result_pid: Result<Pid, _> = child_pid.try_into();

    assert!(child_pid_result_pid.is_ok());

    let child_pid_pid = child_pid_result_pid.unwrap();

    let run_queue_length_after = arc_scheduler.run_queue_len(priority);

    assert_eq!(run_queue_length_after, run_queue_length_before + 1);

    let child_arc_process = pid_to_process(&child_pid_pid).unwrap();

    assert!(scheduler::run_through(&child_arc_process));

    assert_eq!(
        child_arc_process.current_module_function_arity(),
        Some(apply_3::module_function_arity())
    );
    assert_exits_undef(
        &child_arc_process,
        module,
        function,
        arguments,
        // Typo
        ":erlan.+/1 is not exported",
    );

    assert!(!parent_arc_process.is_exiting());
}
