use super::*;

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang;
use crate::test::{assert_exits_badarith, has_message};

#[test]
fn with_valid_arguments_when_run_exits_normal_and_sends_exit_message_to_parent() {
    apply_3::export();

    let parent_arc_process = test::process::init();
    let arc_scheduler = scheduler::current();

    let priority = Priority::Normal;
    let run_queue_length_before = arc_scheduler.run_queue_len(priority);

    erlang::number_or_badarith_1::export();

    let module_atom = erlang::module();
    let module: Term = module_atom.encode().unwrap();

    let function_atom = erlang::number_or_badarith_1::function();
    let function: Term = function_atom.encode().unwrap();

    let number = parent_arc_process.integer(0).unwrap();
    let arguments = parent_arc_process.cons(number, Term::NIL).unwrap();

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

    assert!(scheduler::run_through(&child_arc_process));

    let reason = atom!("normal");

    match *child_arc_process.status.read() {
        Status::Exiting(ref runtime_exception) => {
            assert_eq!(runtime_exception, &exit!(reason, anyhow!("Test").into()));
        }
        ref status => panic!("Process status ({:?}) is not exiting.", status),
    };

    assert!(!parent_arc_process.is_exiting());

    let tag = atom!("DOWN");

    assert_has_message!(
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
    );
}

#[test]
fn without_valid_arguments_when_run_exits_and_sends_parent_exit_message() {
    apply_3::export();

    let parent_arc_process = test::process::init();
    let arc_scheduler = scheduler::current();

    let priority = Priority::Normal;
    let run_queue_length_before = arc_scheduler.run_queue_len(priority);

    erlang::number_or_badarith_1::export();

    let module_atom = erlang::module();
    let module: Term = module_atom.encode().unwrap();

    let function_atom = erlang::number_or_badarith_1::function();
    let function: Term = function_atom.encode().unwrap();

    // not a number
    let number = atom!("zero");
    let arguments = parent_arc_process.cons(number, Term::NIL).unwrap();

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

    assert!(scheduler::run_through(&child_arc_process));

    assert_eq!(
        child_arc_process.current_module_function_arity(),
        Some(Arc::new(ModuleFunctionArity {
            module: atom_from!(module),
            function: atom_from!(function),
            arity: 1
        }))
    );
    assert_exits_badarith(
        &child_arc_process,
        "number (:'zero') is not an integer or a float",
    );

    assert!(!parent_arc_process.is_exiting());

    let tag = atom!("DOWN");
    let reason = atom!("badarith");

    assert_has_message!(
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
    );
}
