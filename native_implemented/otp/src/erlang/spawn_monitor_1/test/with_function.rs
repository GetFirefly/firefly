mod with_arity_zero;

use super::*;

#[test]
fn without_arity_zero_returns_pid_to_parent_and_child_process_exits_badarity_and_sends_exit_message_to_parent(
) {
    let parent_arc_process = test::process::init();
    let module = Atom::from_str("module");
    let function = Atom::from_str("function");
    let arity = 1;

    assert_ne!(arity, 0);

    let function = parent_arc_process.export_closure(module, function, arity, None);

    let result = result(&parent_arc_process, function);

    assert!(result.is_ok());

    let result_boxed_tuple: Result<Boxed<Tuple>, _> = result.unwrap().try_into();

    assert!(result_boxed_tuple.is_ok());

    let boxed_tuple = result_boxed_tuple.unwrap();

    assert_eq!(boxed_tuple.len(), 2);

    let child_pid_term = boxed_tuple[0];

    assert!(child_pid_term.is_pid());

    let child_pid: Pid = child_pid_term.try_into().unwrap();
    let child_arc_process = pid_to_process(&child_pid).unwrap();

    let monitor_reference = boxed_tuple[1];

    assert!(monitor_reference.is_reference());

    let scheduler = scheduler::current();

    assert!(scheduler.run_once());
    assert!(scheduler.run_once());

    assert_exits_badarity(&child_arc_process, function, arity, Term::NIL);

    assert!(!parent_arc_process.is_exiting());

    let tag = atom!("DOWN");
    let reason = badarity_reason(&parent_arc_process, function, Term::NIL);

    assert!(has_message(
        &parent_arc_process,
        parent_arc_process.tuple_from_slice(&[
            tag,
            monitor_reference,
            atom!("process"),
            child_pid_term,
            reason
        ])
    ));
}
