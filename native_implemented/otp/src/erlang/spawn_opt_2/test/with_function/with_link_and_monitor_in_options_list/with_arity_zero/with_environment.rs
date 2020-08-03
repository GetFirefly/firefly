use super::*;

#[test]
fn without_expected_exit_in_child_process_sends_exit_message_to_parent() {
    extern "C" fn native(first: Term, second: Term) -> Term {
        let arc_process = current_process();
        arc_process.reduce();

        fn result(process: &Process, first: Term, second: Term) -> exception::Result<Term> {
            let reason = process.list_from_slice(&[first, second])?;

            Err(exit!(reason, anyhow!("Test").into()).into())
        }

        arc_process.return_status(result(&arc_process, first, second))
    }

    let parent_arc_process = test::process::init();
    let module = Atom::from_str("module");
    let index = Default::default();
    let old_unique = Default::default();
    let unique = Default::default();
    let arity = 0;
    let creator = parent_arc_process.pid().into();

    let function = parent_arc_process
        .anonymous_closure_with_env_from_slice(
            module,
            index,
            old_unique,
            unique,
            arity,
            Some(native as _),
            creator,
            &[Atom::str_to_term("first"), Atom::str_to_term("second")],
        )
        .unwrap();
    let result = result(&parent_arc_process, function, options(&parent_arc_process));

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
        .list_from_slice(&[Atom::str_to_term("first"), Atom::str_to_term("second")])
        .unwrap();

    match *child_arc_process.status.read() {
        Status::RuntimeException(ref exception) => {
            assert_eq!(exception, &exit!(reason, anyhow!("Test").into()));
        }
        ref status => panic!("Child process did not exit.  Status is {:?}", status),
    }

    match *parent_arc_process.status.read() {
        Status::RuntimeException(ref exception) => {
            assert_eq!(
                exception,
                &exit!(
                    child_arc_process
                        .list_from_slice(&[Atom::str_to_term("first"), Atom::str_to_term("second")])
                        .unwrap(),
                    anyhow!("Test").into()
                )
            );
        }
        ref status => panic!("Parent process did not exit.  Status is {:?}", status),
    }

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
fn with_expected_exit_in_child_process_send_exit_message_to_parent() {
    extern "C" fn native(first: Term, second: Term) -> Term {
        let arc_process = current_process();
        arc_process.reduce();

        fn result(process: &Process, first: Term, second: Term) -> exception::Result<Term> {
            let reason = process.tuple_from_slice(&[first, second])?;

            Err(exit!(reason, anyhow!("Test").into()).into())
        }

        arc_process.return_status(result(&arc_process, first, second))
    }

    let parent_arc_process = test::process::init();
    let module = Atom::from_str("module");
    let index = Default::default();
    let old_unique = Default::default();
    let unique = Default::default();
    let arity = 0;
    let creator = parent_arc_process.pid().into();

    let function = parent_arc_process
        .anonymous_closure_with_env_from_slice(
            module,
            index,
            old_unique,
            unique,
            arity,
            Some(native as _),
            creator,
            &[
                Atom::str_to_term("shutdown"),
                Atom::str_to_term("shutdown_reason"),
            ],
        )
        .unwrap();
    let result = result(&parent_arc_process, function, options(&parent_arc_process));

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

    let reason = child_arc_process
        .tuple_from_slice(&[
            Atom::str_to_term("shutdown"),
            Atom::str_to_term("shutdown_reason"),
        ])
        .unwrap();

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
