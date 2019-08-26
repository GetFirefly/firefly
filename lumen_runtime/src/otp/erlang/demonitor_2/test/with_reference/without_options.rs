mod with_monitor;

use super::*;

use crate::otp::erlang::monitor_2;
use crate::process::SchedulerDependentAlloc;

#[test]
fn without_monitor_returns_true() {
    with_process_arc(|monitoring_arc_process| {
        let reference = monitoring_arc_process.next_reference().unwrap();

        assert_eq!(
            native(&monitoring_arc_process, reference, OPTIONS),
            Ok(true.into())
        )
    });
}

const OPTIONS: Term = Term::NIL;
