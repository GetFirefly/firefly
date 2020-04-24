mod with_monitor;

use super::*;

use crate::erlang::monitor_2;
use crate::runtime::scheduler::SchedulerDependentAlloc;

#[test]
fn without_monitor_returns_true() {
    with_process_arc(|monitoring_arc_process| {
        let reference = monitoring_arc_process.next_reference().unwrap();

        assert_eq!(
            result(
                &monitoring_arc_process,
                reference,
                options(&monitoring_arc_process)
            ),
            Ok(true.into())
        )
    });
}

fn options(process: &Process) -> Term {
    process
        .list_from_slice(&[Atom::str_to_term("flush")])
        .unwrap()
}
