mod without_self_pid;

use super::*;

#[test]
fn with_self_pid_sets_group_leader() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::process(),
                Just(arc_process.pid_term()),
            )
        },
        |(arc_process, group_leader_arc_pid, pid)| {
            let group_leader = group_leader_arc_pid.pid_term();

            prop_assert_eq!(result(&arc_process, group_leader, pid), Ok(true.into()));
            prop_assert_eq!(group_leader_0::result(&arc_process), group_leader);

            Ok(())
        }
    );
}
