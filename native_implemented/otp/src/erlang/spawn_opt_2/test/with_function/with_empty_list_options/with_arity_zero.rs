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

    let function = arc_process.export_closure(module, function, arity, NonNull::new(native as _));
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
    };
}

#[test]
fn with_environment_runs_function_in_child_process() {
    extern "C" fn native(function: Term) -> Term {
        let arc_process = current_process();
        arc_process.reduce();

        fn result(process: &Process, function: Term) -> exception::Result<Term> {
            let function_boxed_closure: Boxed<Closure> = function.try_into().unwrap();
            let reason = process.list_from_slice(function_boxed_closure.env_slice());

            Err(exit!(reason, anyhow!("Test").into()).into())
        }

        arc_process.return_status(result(&arc_process, function))
    }

    let module = Atom::from_str("with_environment_runs_function_in_child_process");
    let index = Default::default();
    let old_unique = Default::default();
    let unique = Default::default();

    let parent_arc_process = process::default();
    let creator = parent_arc_process.pid().into();
    let arity = 0;

    let function = parent_arc_process.anonymous_closure_with_env_from_slice(
        module,
        index,
        old_unique,
        unique,
        arity,
        NonNull::new(native as _),
        creator,
        &[Atom::str_to_term("first"), Atom::str_to_term("second")],
    );

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
                    child_arc_process.list_from_slice(&[
                        Atom::str_to_term("first"),
                        Atom::str_to_term("second")
                    ]),
                    anyhow!("Test").into()
                )
            );
        }
        ref status => panic!("Child process did not exit.  Status is {:?}", status),
    };
}
