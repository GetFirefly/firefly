use super::*;

use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::{Boxed, Tuple};

use crate::test::has_message;

#[test]
fn with_arity_when_run_exits_normal_and_sends_exit_message_to_parent() {
    let parent_arc_process = process::test_init();
    let arc_scheduler = Scheduler::current();

    let priority = Priority::Normal;
    let run_queue_length_before = arc_scheduler.run_queue_len(priority);

    let module_atom = Atom::try_from_str("erlang").unwrap();
    let module = unsafe { module_atom.as_term() };

    let function_atom = Atom::try_from_str("self").unwrap();
    let function = unsafe { function_atom.as_term() };

    let arguments = Term::NIL;

    let result = native(&parent_arc_process, module, function, arguments);

    assert!(result.is_ok());

    let returned = result.unwrap();
    let result_boxed_tuple: Result<Boxed<Tuple>, _> = returned.try_into();

    assert!(result_boxed_tuple.is_ok());

    let boxed_tuple: Boxed<Tuple> = result_boxed_tuple.unwrap();

    let child_pid_term = boxed_tuple[0];
    let child_result_pid: Result<Pid, _> = child_pid_term.try_into();

    assert!(child_result_pid.is_ok());

    let child_pid = child_result_pid.unwrap();

    let monitor_reference = boxed_tuple[1];

    assert!(monitor_reference.is_reference());

    let run_queue_length_after = arc_scheduler.run_queue_len(priority);

    assert_eq!(run_queue_length_after, run_queue_length_before + 1);

    let child_arc_process = pid_to_process(&child_pid).unwrap();

    assert!(!parent_arc_process.is_exiting());
    assert!(arc_scheduler.run_through(&child_arc_process));

    assert_eq!(child_arc_process.code_stack_len(), 0);
    assert_eq!(child_arc_process.current_module_function_arity(), None);

    let reason = atom_unchecked("normal");

    match *child_arc_process.status.read() {
        Status::Exiting(ref runtime_exception) => {
            assert_eq!(runtime_exception, &exit!(reason));
        }
        ref status => panic!("Process status ({:?}) is not exiting.", status),
    };

    assert!(!parent_arc_process.is_exiting());

    let tag = atom_unchecked("DOWN");

    assert!(has_message(
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
}

#[test]
fn without_arity_when_run_exits_undef_and_send_exit_message_to_parent() {
    let parent_arc_process = process::test_init();
    let arc_scheduler = Scheduler::current();

    let priority = Priority::Normal;
    let run_queue_length_before = arc_scheduler.run_queue_len(priority);

    let module_atom = Atom::try_from_str("erlang").unwrap();
    let module = unsafe { module_atom.as_term() };

    let function_atom = Atom::try_from_str("+").unwrap();
    let function = unsafe { function_atom.as_term() };

    // `+` is arity 1, not 0
    let arguments = Term::NIL;

    let result = native(&parent_arc_process, module, function, arguments);

    assert!(result.is_ok());

    let returned = result.unwrap();
    let result_boxed_tuple: Result<Boxed<Tuple>, _> = returned.try_into();

    assert!(result_boxed_tuple.is_ok());

    let boxed_tuple: Boxed<Tuple> = result_boxed_tuple.unwrap();

    let child_pid_term = boxed_tuple[0];
    let child_result_pid: Result<Pid, _> = child_pid_term.try_into();

    assert!(child_result_pid.is_ok());

    let child_pid = child_result_pid.unwrap();

    let monitor_reference = boxed_tuple[1];

    assert!(monitor_reference.is_reference());

    let run_queue_length_after = arc_scheduler.run_queue_len(priority);

    assert_eq!(run_queue_length_after, run_queue_length_before + 1);

    let child_arc_process = pid_to_process(&child_pid).unwrap();

    assert!(arc_scheduler.run_through(&child_arc_process));

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

    let tag = atom_unchecked("DOWN");
    let reason = match undef!(&parent_arc_process, module, function, arguments) {
        Exception::Runtime(runtime_exception) => runtime_exception.reason,
        _ => unreachable!("parent process out-of-memory"),
    };

    assert!(has_message(
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
}
