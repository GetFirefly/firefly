mod with_registered_name;

use super::*;

#[test]
fn without_registered_name_returns_reference_but_immediate_sends_noproc_message() {
    with_process_arc(|monitoring_arc_process| {
        let registered_name = registered_name();

        let monitor_reference_result = result(&monitoring_arc_process, r#type(), registered_name);

        assert!(monitor_reference_result.is_ok());

        let monitor_reference = monitor_reference_result.unwrap();

        assert!(monitor_reference.is_reference());

        let tag = Atom::str_to_term("DOWN");
        let reason = Atom::str_to_term("noproc");

        assert_has_message!(
            &monitoring_arc_process,
            monitoring_arc_process
                .tuple_from_slice(&[
                    tag,
                    monitor_reference,
                    r#type(),
                    monitoring_arc_process
                        .tuple_from_slice(&[registered_name, node_0::result()])
                        .unwrap(),
                    reason
                ])
                .unwrap()
        );
    });
}
