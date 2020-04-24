use super::*;

#[test]
fn without_environment_runs_function_in_child_process() {
    let module = Atom::from_str("module");
    let function = Atom::from_str("function");
    let arc_process = process::default();
    let arity = 0;

    fn native_result() -> exception::Result<Term> {
        Ok(Atom::str_to_term("ok"))
    }

    extern "C" fn native() -> Term {
        let arc_process = current_process();
        arc_process.reduce();

        arc_process.return_status(native_result())
    }

    let function = arc_process
        .export_closure(module, function, arity, Some(native as _))
        .unwrap();
    let result = result(&arc_process, function, options(&arc_process));

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
                &exit!(Atom::str_to_term("normal"), anyhow!("Test").into())
            );
        }
        ref status => panic!("Child process did not exit.  Status is {:?}", status),
    }

    std::mem::drop(child_arc_process);
}

#[test]
fn with_environment_runs_function_in_child_process() {
    let module = Atom::from_str("module");
    let index = Default::default();
    let old_unique = Default::default();
    let unique = Default::default();

    let arc_process = process::default();
    let creator = arc_process.pid().into();
    let arity = 0;

    fn native_result(process: &Process, first: Term, second: Term) -> exception::Result<Term> {
        let reason = process.list_from_slice(&[first, second])?;

        Err(exit!(reason, anyhow!("Test").into()).into())
    }

    extern "C" fn native(first: Term, second: Term) -> Term {
        let arc_process = current_process();
        arc_process.reduce();

        arc_process.return_status(native_result(&arc_process, first, second))
    }

    let function = arc_process
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
    let result = result(&arc_process, function, options(&arc_process));

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
                        .list_from_slice(&[Atom::str_to_term("first"), Atom::str_to_term("second")])
                        .unwrap(),
                    anyhow!("Test").into()
                )
            );
        }
        ref status => panic!("Child process did not exit.  Status is {:?}", status),
    }

    std::mem::drop(child_arc_process);
}
