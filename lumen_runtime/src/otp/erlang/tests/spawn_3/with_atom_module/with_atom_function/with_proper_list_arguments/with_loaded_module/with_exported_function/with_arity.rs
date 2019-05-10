use super::*;

#[test]
fn with_valid_arguments_when_run_returns() {
    with_process(|parent_process| {
        let arc_scheduler = Scheduler::current();
        let arc_mutex_run_queue = arc_scheduler.run_queue(Priority::Normal);

        let run_queue_length_before = arc_mutex_run_queue.lock().unwrap().len();

        let module = Term::str_to_atom("erlang", DoNotCare).unwrap();
        let function = Term::str_to_atom("+", DoNotCare).unwrap();
        let number = 0.into_process(parent_process);
        let arguments = Term::cons(number, Term::EMPTY_LIST, parent_process);

        let result = erlang::spawn_3(module, function, arguments, parent_process);

        assert!(result.is_ok());

        let child_pid = result.unwrap();

        assert_eq!(child_pid.tag(), LocalPid);

        let run_queue_length_after = arc_mutex_run_queue.lock().unwrap().len();

        assert_eq!(run_queue_length_after, run_queue_length_before + 1);

        let arc_process = pid_to_process(child_pid).unwrap();

        arc_scheduler.run_through(&arc_process);

        let stack = arc_process.stack.lock().unwrap();

        assert_eq!(stack.len(), 1);
        assert_eq!(stack.get(0).unwrap(), &number);

        match *arc_process.status.read().unwrap() {
            Status::Exiting(ref exception) => {
                assert_eq!(
                    exception,
                    &exit!(Term::str_to_atom("normal", DoNotCare).unwrap())
                );
            }
            ref status => panic!("Process status ({:?}) is not exiting.", status),
        };
    });
}

#[test]
fn without_valid_arguments_when_run_exits() {
    with_process(|parent_process| {
        let arc_scheduler = Scheduler::current();
        let arc_mutex_run_queue = arc_scheduler.run_queue(Priority::Normal);

        let run_queue_length_before = arc_mutex_run_queue.lock().unwrap().len();

        let module = Term::str_to_atom("erlang", DoNotCare).unwrap();
        let function = Term::str_to_atom("+", DoNotCare).unwrap();
        // not a number
        let number = Term::str_to_atom("zero", DoNotCare).unwrap();
        let arguments = Term::cons(number, Term::EMPTY_LIST, parent_process);

        let result = erlang::spawn_3(module, function, arguments, parent_process);

        assert!(result.is_ok());

        let child_pid = result.unwrap();

        assert_eq!(child_pid.tag(), LocalPid);

        let run_queue_length_after = arc_mutex_run_queue.lock().unwrap().len();

        assert_eq!(run_queue_length_after, run_queue_length_before + 1);

        let arc_process = pid_to_process(child_pid).unwrap();

        arc_scheduler.run_through(&arc_process);

        let stack = arc_process.stack.lock().unwrap();

        assert_eq!(stack.len(), 0);

        match *arc_process.status.read().unwrap() {
            Status::Exiting(ref exception) => {
                assert_eq!(exception, &badarith!());
            }
            ref status => panic!("Process status ({:?}) is not exiting.", status),
        };
    });
}
