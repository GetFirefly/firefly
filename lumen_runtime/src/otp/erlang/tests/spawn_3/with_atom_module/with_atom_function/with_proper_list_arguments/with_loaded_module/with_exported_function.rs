use super::*;

mod with_arity;

#[test]
fn without_arity_when_run_exits_undef() {
    with_process(|parent_process| {
        let arc_scheduler = Scheduler::current();

        let priority = Priority::Normal;
        let run_queue_length_before = arc_scheduler.run_queue_len(priority);

        let module = Term::str_to_atom("erlang", DoNotCare).unwrap();
        let function = Term::str_to_atom("+", DoNotCare).unwrap();
        // erlang.+/1 and erlang.+/2 exists so use 3 for invalid arity
        let arguments = Term::slice_to_list(
            &[
                0.into_process(parent_process),
                1.into_process(parent_process),
                2.into_process(parent_process),
            ],
            parent_process,
        );

        let result = erlang::spawn_3(module, function, arguments, parent_process);

        assert!(result.is_ok());

        let child_pid = result.unwrap();

        assert_eq!(child_pid.tag(), LocalPid);

        let run_queue_length_after = arc_scheduler.run_queue_len(priority);

        assert_eq!(run_queue_length_after, run_queue_length_before + 1);

        let arc_process = pid_to_process(child_pid).unwrap();

        arc_scheduler.run_through(&arc_process);

        assert_eq!(arc_process.stack_len(), 1);
        assert_eq!(
            arc_process.current_module_function_arity(),
            Some(Arc::new(ModuleFunctionArity {
                module,
                function,
                arity: 3
            }))
        );

        match *arc_process.status.read().unwrap() {
            Status::Exiting(ref exception) => {
                assert_eq!(
                    exception,
                    &undef!(module, function, arguments, &arc_process)
                );
            }
            ref status => panic!("Process status ({:?}) is not exiting.", status),
        };
    });
}
