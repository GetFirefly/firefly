use super::*;

#[test]
fn without_expected_exit_in_child_process_exits_linked_parent_process() {
    let parent_arc_process = test::process::init();

    let creator = parent_arc_process.pid().into();
    let arity = 0;
    let function = parent_arc_process.anonymous_closure_with_env_from_slice(
        Atom::from_str("without_expected_exit_in_child_process_exits_linked_parent_process"),
        Default::default(),
        Default::default(),
        Default::default(),
        arity,
        NonNull::new(native as _),
        creator,
        &[Atom::str_to_term("first"), Atom::str_to_term("second")],
    );

    let result = result(&parent_arc_process, function);

    assert!(result.is_ok());

    let child_pid_term = result.unwrap();

    assert!(child_pid_term.is_pid());

    let child_pid: Pid = child_pid_term.try_into().unwrap();

    let child_arc_process = pid_to_process(&child_pid).unwrap();

    let scheduler = scheduler::current();

    assert!(scheduler.run_once());
    assert!(scheduler.run_once());

    match *child_arc_process.status.read() {
        Status::RuntimeException(ref exception) => {
            assert_eq!(
                exception,
                &exit!(
                    child_arc_process.tuple_from_slice(&[
                        Atom::str_to_term("first"),
                        Atom::str_to_term("second")
                    ]),
                    anyhow!("Test").into()
                )
            );
        }
        ref status => panic!("Child process did not exit.  Status is {:?}", status),
    }

    match *parent_arc_process.status.read() {
        Status::RuntimeException(ref exception) => {
            assert_eq!(
                exception,
                &exit!(
                    child_arc_process.tuple_from_slice(&[
                        Atom::str_to_term("first"),
                        Atom::str_to_term("second")
                    ]),
                    anyhow!("Test").into()
                )
            );
        }
        ref status => panic!("Parent process did not exit.  Status is {:?}", status),
    };
}

#[test]
fn with_expected_exit_in_child_process_does_not_exit_linked_parent_process() {
    let parent_arc_process = test::process::init();

    let creator = parent_arc_process.pid().into();
    let arity = 0;
    let function = parent_arc_process.anonymous_closure_with_env_from_slice(
        Atom::from_str("with_expected_exit_in_child_process_does_not_exit_linked_parent_process"),
        Default::default(),
        Default::default(),
        Default::default(),
        arity,
        NonNull::new(native as _),
        creator,
        &[
            Atom::str_to_term("shutdown"),
            Atom::str_to_term("shutdown_reason"),
        ],
    );

    let result = result(&parent_arc_process, function);

    assert!(result.is_ok());

    let child_pid_term = result.unwrap();

    assert!(child_pid_term.is_pid());

    let child_pid: Pid = child_pid_term.try_into().unwrap();

    let child_arc_process = pid_to_process(&child_pid).unwrap();

    let scheduler = scheduler::current();

    assert!(scheduler.run_once());
    assert!(scheduler.run_once());

    match *child_arc_process.status.read() {
        Status::RuntimeException(ref exception) => {
            assert_eq!(
                exception,
                &exit!(
                    child_arc_process.tuple_from_slice(&[
                        Atom::str_to_term("shutdown"),
                        Atom::str_to_term("shutdown_reason")
                    ]),
                    anyhow!("Test").into()
                )
            );
        }
        ref status => panic!("Child process did not exit.  Status is {:?}", status),
    }

    match *parent_arc_process.status.read() {
        Status::RuntimeException(ref exception) => panic!("Parent process exited {:?}", exception),
        _ => (),
    };
}

extern "C" fn native(function: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    fn result(process: &Process, function: Term) -> exception::Result<Term> {
        let function_boxed_closure: Boxed<Closure> = function.try_into().unwrap();
        let reason = process.tuple_from_slice(function_boxed_closure.env_slice());

        Err(exit!(reason, anyhow!("Test").into()).into())
    }

    arc_process.return_status(result(&arc_process, function))
}
