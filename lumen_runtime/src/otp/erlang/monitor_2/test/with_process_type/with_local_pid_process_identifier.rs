mod with_process;

use super::*;

use liblumen_alloc::erts::term::next_pid;

#[test]
fn without_process_returns_reference_but_immediate_sends_noproc_message() {
    with_process_arc(|monitoring_arc_process| {
        let monitored_pid = next_pid();

        let monitor_reference_result = native(&monitoring_arc_process, r#type(), monitored_pid);

        assert!(monitor_reference_result.is_ok());

        let monitor_reference = monitor_reference_result.unwrap();

        assert!(monitor_reference.is_reference());

        let tag = atom_unchecked("DOWN");
        let reason = atom_unchecked("noproc");

        assert!(has_message(
            &monitoring_arc_process,
            monitoring_arc_process
                .tuple_from_slice(&[tag, monitor_reference, r#type(), monitored_pid, reason])
                .unwrap()
        ));
    });
}
