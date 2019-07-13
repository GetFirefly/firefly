use super::*;

#[test]
fn with_valid_arguments_when_run_returns() {
    with_process(|parent_process| {
        let arc_scheduler = Scheduler::current();

        let priority = Priority::Normal;
        let run_queue_length_before = arc_scheduler.run_queue_len(priority);

        let module_atom = Atom::try_from_str("erlang").unwrap();
        let module = unsafe { module_atom.as_term() };

        let function_atom = Atom::try_from_str("+").unwrap();
        let function = unsafe { function_atom.as_term() };

        let number = parent_process.integer(0);
        let arguments = parent_process.cons(number, Term::NIL).unwrap();

        let result = erlang::spawn_3(module, function, arguments, parent_process);

        assert!(result.is_ok());

        let child_pid = result.unwrap();
        let child_pid_result_pid: core::result::Result<Pid, _> = child_pid.try_into();

        assert!(child_pid_result_pid.is_ok());

        let child_pid_pid = child_pid_result_pid.unwrap();

        let run_queue_length_after = arc_scheduler.run_queue_len(priority);

        assert_eq!(run_queue_length_after, run_queue_length_before + 1);

        let arc_process = pid_to_process(child_pid_pid).unwrap();

        arc_scheduler.run_through(&arc_process);

        assert_eq!(arc_process.code_stack_len(), 1);
        assert_eq!(
            arc_process.current_module_function_arity(),
            Some(Arc::new(ModuleFunctionArity {
                module: module_atom,
                function: function_atom,
                arity: 1
            }))
        );

        match *arc_process.status.read() {
            Status::Exiting(ref runtime_exception) => {
                assert_eq!(runtime_exception, &exit!(number));
            }
            ref status => panic!("ProcessControlBlock status ({:?}) is not exiting.", status),
        };
    });
}

#[test]
fn without_valid_arguments_when_run_exits() {
    with_process(|parent_process| {
        let arc_scheduler = Scheduler::current();

        let priority = Priority::Normal;
        let run_queue_length_before = arc_scheduler.run_queue_len(priority);

        let module_atom = Atom::try_from_str("erlang").unwrap();
        let module = unsafe { module_atom.as_term() };

        let function_atom = Atom::try_from_str("+").unwrap();
        let function = unsafe { function_atom.as_term() };

        // not a number
        let number = atom_unchecked("zero");
        let arguments = parent_process.cons(number, Term::NIL).unwrap();

        let result = erlang::spawn_3(module, function, arguments, parent_process);

        assert!(result.is_ok());

        let child_pid = result.unwrap();
        let child_pid_result_pid: core::result::Result<Pid, _> = child_pid.try_into();

        assert!(child_pid_result_pid.is_ok());

        let child_pid_pid = child_pid_result_pid.unwrap();

        let run_queue_length_after = arc_scheduler.run_queue_len(priority);

        assert_eq!(run_queue_length_after, run_queue_length_before + 1);

        let arc_process = pid_to_process(child_pid_pid).unwrap();

        arc_scheduler.run_through(&arc_process);

        assert_eq!(arc_process.code_stack_len(), 1);
        assert_eq!(
            arc_process.current_module_function_arity(),
            Some(Arc::new(ModuleFunctionArity {
                module: module_atom,
                function: function_atom,
                arity: 1
            }))
        );

        match *arc_process.status.read() {
            Status::Exiting(ref runtime_exception) => {
                assert_eq!(runtime_exception, &badarith!());
            }
            ref status => panic!("ProcessControlBlock status ({:?}) is not exiting.", status),
        };
    });
}
