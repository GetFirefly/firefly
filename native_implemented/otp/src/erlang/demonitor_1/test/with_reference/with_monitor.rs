use super::*;

// `returns_true` in integration tests

#[test]
fn does_not_flush_existing_message() {
    with_process_arc(|monitoring_arc_process| {
        let monitored_arc_process = test::process::child(&monitoring_arc_process);
        let monitored_pid_term = monitored_arc_process.pid_term();

        let monitor_reference =
            monitor_2::result(&monitoring_arc_process, r#type(), monitored_pid_term).unwrap();

        let reason = Atom::str_to_term("normal");
        exit_when_run(&monitored_arc_process, reason);

        assert!(scheduler::run_through(&monitored_arc_process));

        assert!(monitored_arc_process.is_exiting());
        assert!(!monitoring_arc_process.is_exiting());

        let tag = Atom::str_to_term("DOWN");

        assert_has_message!(
            &monitoring_arc_process,
            monitoring_arc_process
                .tuple_from_slice(&[tag, monitor_reference, r#type(), monitored_pid_term, reason])
                .unwrap()
        );

        assert_eq!(
            result(&monitoring_arc_process, monitor_reference),
            Ok(true.into())
        );

        assert_has_message!(
            &monitoring_arc_process,
            monitoring_arc_process
                .tuple_from_slice(&[tag, monitor_reference, r#type(), monitored_pid_term, reason])
                .unwrap()
        );
    });
}

// `prevents_future_messages` in integration tests

fn r#type() -> Term {
    Atom::str_to_term("process")
}
