mod with_monitor;

use super::*;

#[test]
fn without_monitor_returns_true() {
    with_process_arc(|monitoring_arc_process| {
        let reference = monitoring_arc_process.next_reference().unwrap();

        assert_eq!(result(&monitoring_arc_process, reference), Ok(true.into()))
    });
}
