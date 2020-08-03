mod with_process;

use super::*;

#[test]
fn without_process_returns_reference_but_immediate_sends_noproc_message() {
    with_process_arc(|monitoring_arc_process| {
        let monitored_pid = Pid::next_term();

        let monitor_reference_result = result(&monitoring_arc_process, r#type(), monitored_pid);

        assert!(monitor_reference_result.is_ok());

        let monitor_reference = monitor_reference_result.unwrap();

        assert!(monitor_reference.is_reference());

        let tag = Atom::str_to_term("DOWN");
        let reason = Atom::str_to_term("noproc");

        assert_has_message!(
            &monitoring_arc_process,
            monitoring_arc_process
                .tuple_from_slice(&[tag, monitor_reference, r#type(), monitored_pid, reason])
                .unwrap()
        );
    });
}
