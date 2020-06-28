use super::*;

mod with_arity;

#[test]
fn without_arity_when_run_exits_undef_and_parent_exits() {
    let parent_arc_process = test::process::init();
    let arc_scheduler = scheduler::current();

    let priority = Priority::Normal;
    let run_queue_length_before = arc_scheduler.run_queue_len(priority);

    let module = atom!("erlang");
    let function = atom!("+");

    // erlang.+/1 and erlang.+/2 exists so use 3 for invalid arity
    let arguments = parent_arc_process
        .list_from_slice(&[
            parent_arc_process.integer(0).unwrap(),
            parent_arc_process.integer(1).unwrap(),
            parent_arc_process.integer(2).unwrap(),
        ])
        .unwrap();

    let result = result(&parent_arc_process, module, function, arguments);

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
        ":erlang.+/3 is not exported",
    );

    assert!(child_arc_process.is_exiting())
}
