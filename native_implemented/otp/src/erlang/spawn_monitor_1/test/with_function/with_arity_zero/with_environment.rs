use super::*;

#[test]
fn without_expected_exit_in_child_process_sends_exit_message_to_parent() {
    let arc_process = test::process::init();
    let creator = arc_process.pid().into();
    let arity = 0;

    fn native(function: Term) -> Term {
        let arc_process = current_process();
        arc_process.reduce();

        fn result(process: &Process, function: Term) -> exception::Result<Term> {
            let function_boxed_closure: Boxed<Closure> = function.try_into().unwrap();
            let reason = process.list_from_slice(function_boxed_closure.env_slice());

            Err(exit!(reason, anyhow!("Test").into()).into())
        }

        arc_process.return_status(result(&arc_process, function))
    }

    let parent_arc_process = arc_process.clone();

    let module =
        Atom::from_str("without_expected_exit_in_child_process_sends_exit_message_to_parent");
    let index = Default::default();
    let old_unique = Default::default();
    let unique = Default::default();
    let function = arc_process.anonymous_closure_with_env_from_slice(
        module,
        index,
        old_unique,
        unique,
        arity,
        NonNull::new(native as _),
        creator,
        &[Atom::str_to_term("first"), Atom::str_to_term("second")],
    );
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

    let reason = child_arc_process
        .list_from_slice(&[Atom::str_to_term("first"), Atom::str_to_term("second")]);

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
        parent_arc_process.tuple_from_slice(&[
            tag,
            monitor_reference,
            Atom::str_to_term("process"),
            child_pid_term,
            reason
        ])
    ));
}

#[test]
fn with_expected_exit_in_child_process_send_exit_message_to_parent() {
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

    let parent_arc_process = test::process::init();
    let arity = 0;
    let creator = parent_arc_process.pid().into();

    let function = parent_arc_process.anonymous_closure_with_env_from_slice(
        Atom::from_str("with_expected_exit_in_child_process_send_exit_message_to_parent"),
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

    let reason = child_arc_process.tuple_from_slice(&[
        Atom::str_to_term("shutdown"),
        Atom::str_to_term("shutdown_reason"),
    ]);

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
        parent_arc_process.tuple_from_slice(&[
            tag,
            monitor_reference,
            Atom::str_to_term("process"),
            child_pid_term,
            reason
        ])
    ));
}