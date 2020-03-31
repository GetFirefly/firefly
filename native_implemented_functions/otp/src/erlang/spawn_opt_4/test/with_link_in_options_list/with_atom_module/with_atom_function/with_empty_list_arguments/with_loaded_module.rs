use super::*;

mod with_exported_function;

#[test]
fn without_exported_function_when_run_exits_undef_and_parent_exits() {
    apply_3::export();

    let parent_arc_process = test::process::init();
    let arc_scheduler = scheduler::current();

    let priority = Priority::Normal;
    let run_queue_length_before = arc_scheduler.run_queue_len(priority);

    let module = atom!("erlang");
    // Typo
    let function = atom!("sel");

    let arguments = Term::NIL;

    let result = native(
        &parent_arc_process,
        module,
        function,
        arguments,
        options(&parent_arc_process),
    );

    assert!(result.is_ok());

    let child_pid = result.unwrap();
    let child_pid_result_pid: Result<Pid, _> = child_pid.try_into();

    assert!(child_pid_result_pid.is_ok());

    let child_pid_pid = child_pid_result_pid.unwrap();

    let run_queue_length_after = arc_scheduler.run_queue_len(priority);

    assert_eq!(run_queue_length_after, run_queue_length_before + 1);

    let arc_process = pid_to_process(&child_pid_pid).unwrap();

    assert!(scheduler::run_through(&arc_process));
    assert!(!scheduler::run_through(&arc_process));

    assert_eq!(
        arc_process.current_module_function_arity(),
        Some(apply_3::module_function_arity())
    );
    assert_exits_undef(
        &arc_process,
        module,
        function,
        arguments,
        ":erlang.sel/0 is not exported",
    );
}
