use super::*;
use lumen_rt_full::process::current_process;

#[test]
fn without_expected_exit_in_child_process_sends_exit_message_to_parent() {
    let parent_arc_process = test::process::init();
    let module = Atom::from_str("module");
    let function = Atom::from_str("function");
    let arity = 0;

    extern "C" fn native() -> Term {
        let arc_process = current_process();
        arc_process.reduce();

        fn result() -> exception::Result<Term> {
            Err(exit!(Atom::str_to_term("not_normal"), anyhow!("Test").into()).into())
        }

        arc_process.return_status(result())
    }

    let function = parent_arc_process
        .export_closure(module, function, arity, Some(native as _))
        .unwrap();
    let result = result(&parent_arc_process, function);

    assert!(result.is_ok());

    let returned = result.unwrap();

    let result_boxed_tuple: Result<Boxed<Tuple>, _> = returned.try_into();

    assert!(
        result_boxed_tuple.is_ok(),
        "Returned ({:?}) is not a tuple",
        returned
    );

    let boxed_tuple = result_boxed_tuple.unwrap();

    assert_eq!(boxed_tuple.len(), 2);

    let child_pid_term = boxed_tuple[0];

    assert!(child_pid_term.is_pid());

    let child_pid: Pid = child_pid_term.try_into().unwrap();
    let child_arc_process = pid_to_process(&child_pid).unwrap();

    let monitor_reference = boxed_tuple[1];

    assert!(monitor_reference.is_reference());

    assert!(scheduler::run_through(&child_arc_process));

    let reason = Atom::str_to_term("not_normal");

    match *child_arc_process.status.read() {
        Status::RuntimeException(ref exception) => {
            assert_eq!(exception, &exit!(reason, anyhow!("Test").into()));
        }
        ref status => panic!("Child process did not exit.  Status is {:?}", status),
    }

    assert!(!parent_arc_process.is_exiting());

    let tag = Atom::str_to_term("DOWN");

    assert!(has_message(
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
}

#[test]
fn with_expected_exit_in_child_process_sends_exit_message_to_parent() {
    extern "C" fn native() -> Term {
        let arc_process = current_process();
        arc_process.reduce();

        fn result() -> exception::Result<Term> {
            Ok(Atom::str_to_term("ok"))
        }

        arc_process.return_status(result())
    }

    let parent_arc_process = test::process::init();
    let module = Atom::from_str("module");
    let function = Atom::from_str("function");
    let arity = 0;
    let function = parent_arc_process
        .export_closure(module, function, arity, Some(native as _))
        .unwrap();
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
    assert!(scheduler::run_through(&child_arc_process));

    let reason = Atom::str_to_term("normal");

    match *child_arc_process.status.read() {
        Status::RuntimeException(ref exception) => {
            assert_eq!(exception, &exit!(reason, anyhow!("Test").into()));
        }
        ref status => panic!("Child process did not exit.  Status is {:?}", status),
    }

    assert!(!parent_arc_process.is_exiting());

    let tag = Atom::str_to_term("DOWN");

    assert!(has_message(
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
}
