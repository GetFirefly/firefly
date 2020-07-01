use super::*;

#[test]
fn returns_true() {
    with_process_arc(|monitoring_arc_process| {
        let monitored_arc_process = test::process::child(&monitoring_arc_process);

        let monitor_reference = monitor_2::result(
            &monitoring_arc_process,
            r#type(),
            monitored_arc_process.pid_term(),
        )
        .unwrap();

        let monitored_monitor_count_before = monitor_count(&monitored_arc_process);
        let monitoring_monitored_count_before = monitored_count(&monitoring_arc_process);

        assert_eq!(
            result(&monitoring_arc_process, monitor_reference),
            Ok(true.into())
        );

        let monitored_monitor_count_after = monitor_count(&monitored_arc_process);
        let monitoring_monitored_count_after = monitored_count(&monitoring_arc_process);

        assert_eq!(
            monitored_monitor_count_after,
            monitored_monitor_count_before - 1
        );
        assert_eq!(
            monitoring_monitored_count_after,
            monitoring_monitored_count_before - 1
        );
    });
}

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

#[test]
fn prevents_future_messages() {
    with_process_arc(|monitoring_arc_process| {
        let monitored_arc_process = test::process::child(&monitoring_arc_process);
        let monitored_pid_term = monitored_arc_process.pid_term();

        let monitor_reference =
            monitor_2::result(&monitoring_arc_process, r#type(), monitored_pid_term).unwrap();

        let reason = Atom::str_to_term("normal");
        let tag = Atom::str_to_term("DOWN");

        assert!(!has_message(
            &monitoring_arc_process,
            monitoring_arc_process
                .tuple_from_slice(&[tag, monitor_reference, r#type(), monitored_pid_term, reason])
                .unwrap()
        ));

        assert_eq!(
            result(&monitoring_arc_process, monitor_reference),
            Ok(true.into())
        );

        exit_when_run(&monitored_arc_process, reason);

        assert!(scheduler::run_through(&monitored_arc_process));

        assert!(monitored_arc_process.is_exiting());
        assert!(!monitoring_arc_process.is_exiting());

        assert!(!has_message(
            &monitoring_arc_process,
            monitoring_arc_process
                .tuple_from_slice(&[tag, monitor_reference, r#type(), monitored_pid_term, reason])
                .unwrap()
        ));
    });
}

fn r#type() -> Term {
    Atom::str_to_term("process")
}
