use super::*;

#[test]
fn without_expected_exit_in_child_process_exits_linked_parent_process() {
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
    let creator = parent_arc_process.pid().into();
    let arity = 0;
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
                    child_arc_process
                        .list_from_slice(&[Atom::str_to_term("first"), Atom::str_to_term("second")])
                        .unwrap(),
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
                    child_arc_process
                        .list_from_slice(&[Atom::str_to_term("first"), Atom::str_to_term("second")])
                        .unwrap(),
                    anyhow!("Test").into()
                )
            );
        }
        ref status => panic!("Parent process did not exit.  Status is {:?}", status),
    }

    std::mem::drop(parent_arc_process)
}

#[test]
fn with_expected_exit_in_child_process_does_not_exit_linked_parent_process() {
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

    let child_pid_term = result.unwrap();

    assert!(child_pid_term.is_pid());

    let child_pid: Pid = child_pid_term.try_into().unwrap();
    let child_arc_process = pid_to_process(&child_pid).unwrap();

    assert!(scheduler::run_through(&child_arc_process));

    match *child_arc_process.status.read() {
        Status::RuntimeException(ref exception) => {
            assert_eq!(
                exception,
                &exit!(
                    child_arc_process
                        .tuple_from_slice(&[
                            Atom::str_to_term("shutdown"),
                            Atom::str_to_term("shutdown_reason")
                        ])
                        .unwrap(),
                    anyhow!("Test").into()
                )
            );
        }
        ref status => panic!("Child process did not exit.  Status is {:?}", status),
    }

    match *parent_arc_process.status.read() {
        Status::RuntimeException(ref exception) => panic!("Parent process exited {:?}", exception),
        _ => (),
    }

    std::mem::drop(parent_arc_process);
}
