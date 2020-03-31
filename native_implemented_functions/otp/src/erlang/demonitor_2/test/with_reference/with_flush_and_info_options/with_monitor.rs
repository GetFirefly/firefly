use super::*;

#[test]
fn returns_true() {
    with_monitor_returns_true(options);
}

#[test]
fn flushes_existing_message_and_returns_false() {
    with_process_arc(|monitoring_arc_process| {
        let monitored_arc_process = test::process::child(&monitoring_arc_process);
        let monitored_pid_term = monitored_arc_process.pid_term();

        let monitor_reference =
            monitor_2::native(&monitoring_arc_process, r#type(), monitored_pid_term).unwrap();

        let reason = Atom::str_to_term("normal");
        exit_1::place_frame_with_arguments(&monitored_arc_process, Placement::Replace, reason)
            .unwrap();

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
            native(
                &monitoring_arc_process,
                monitor_reference,
                options(&monitoring_arc_process)
            ),
            Ok(false.into())
        );

        assert!(!has_message(
            &monitoring_arc_process,
            monitoring_arc_process
                .tuple_from_slice(&[tag, monitor_reference, r#type(), monitored_pid_term, reason])
                .unwrap()
        ));
    });
}

#[test]
fn prevents_future_messages() {
    super::prevents_future_messages(options);
}
