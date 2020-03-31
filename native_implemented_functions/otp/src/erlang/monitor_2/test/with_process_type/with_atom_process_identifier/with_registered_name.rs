use super::*;

#[test]
fn returns_reference() {
    with_process_arc(|monitoring_arc_process| {
        let registered_name = registered_name();
        let registered_name_atom: Atom = registered_name.try_into().unwrap();

        let monitored_arc_process = test::process::child(&monitoring_arc_process);

        assert!(registry::put_atom_to_process(
            registered_name_atom,
            monitored_arc_process.clone()
        ));

        let monitored_monitor_count_before = monitor_count(&monitored_arc_process);
        let monitoring_monitored_count_before = monitored_count(&monitoring_arc_process);

        let monitor_reference_result = native(&monitoring_arc_process, r#type(), registered_name);

        assert!(monitor_reference_result.is_ok());

        let monitor_reference = monitor_reference_result.unwrap();

        assert!(monitor_reference.is_reference());

        let monitored_monitor_count_after = monitor_count(&monitored_arc_process);
        let monitoring_monitored_count_after = monitored_count(&monitoring_arc_process);

        assert_eq!(
            monitored_monitor_count_after,
            monitored_monitor_count_before + 1
        );
        assert_eq!(
            monitoring_monitored_count_after,
            monitoring_monitored_count_before + 1
        );
    });
}

#[test]
fn returns_different_reference_each_time() {
    with_process_arc(|monitoring_arc_process| {
        let registered_name = registered_name();
        let registered_name_atom: Atom = registered_name.try_into().unwrap();

        let monitored_arc_process = test::process::child(&monitoring_arc_process);

        assert!(registry::put_atom_to_process(
            registered_name_atom,
            monitored_arc_process.clone()
        ));

        let monitored_monitor_count_before = monitor_count(&monitored_arc_process);
        let monitoring_monitored_count_before = monitored_count(&monitoring_arc_process);

        let first_monitor_reference =
            native(&monitoring_arc_process, r#type(), registered_name).unwrap();
        let second_monitor_reference =
            native(&monitoring_arc_process, r#type(), registered_name).unwrap();

        assert_ne!(first_monitor_reference, second_monitor_reference);

        let monitored_monitor_count_after = monitor_count(&monitored_arc_process);
        let monitoring_monitored_count_after = monitored_count(&monitoring_arc_process);

        assert_eq!(
            monitored_monitor_count_after,
            monitored_monitor_count_before + 2
        );
        assert_eq!(
            monitoring_monitored_count_after,
            monitoring_monitored_count_before + 2
        );
    });
}

#[test]
fn when_monitored_process_exits_it_sends_message_for_each_monitor_reference() {
    with_process_arc(|monitoring_arc_process| {
        let monitored_arc_process = test::process::child(&monitoring_arc_process);

        let first_registered_name = registered_name();
        let first_registered_name_atom: Atom = first_registered_name.try_into().unwrap();

        assert!(registry::put_atom_to_process(
            first_registered_name_atom,
            monitored_arc_process.clone()
        ));

        let first_monitor_reference =
            native(&monitoring_arc_process, r#type(), first_registered_name).unwrap();

        let second_registered_name = registered_name();
        let second_registered_name_atom: Atom = second_registered_name.try_into().unwrap();

        assert!(registry::unregister(&first_registered_name_atom));
        assert!(registry::put_atom_to_process(
            second_registered_name_atom,
            monitored_arc_process.clone()
        ));

        let second_monitor_reference =
            native(&monitoring_arc_process, r#type(), second_registered_name).unwrap();

        assert!(!monitored_arc_process.is_exiting());

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
                .tuple_from_slice(&[
                    tag,
                    first_monitor_reference,
                    r#type(),
                    monitoring_arc_process
                        .tuple_from_slice(&[first_registered_name, node_0::native()])
                        .unwrap(),
                    reason
                ])
                .unwrap()
        );
        assert_has_message!(
            &monitoring_arc_process,
            monitoring_arc_process
                .tuple_from_slice(&[
                    tag,
                    second_monitor_reference,
                    r#type(),
                    monitoring_arc_process
                        .tuple_from_slice(&[second_registered_name, node_0::native()])
                        .unwrap(),
                    reason
                ])
                .unwrap()
        );
    });
}
