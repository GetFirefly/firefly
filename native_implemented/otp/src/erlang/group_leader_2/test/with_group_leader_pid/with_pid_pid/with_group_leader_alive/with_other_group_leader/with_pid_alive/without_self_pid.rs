use super::*;

#[test]
fn with_different_group_leader_and_pid_sets_group_leader() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::process(),
                strategy::process(),
            )
        },
        |(arc_process, group_leader_arc_process, pid_arc_process)| {
            let group_leader = group_leader_arc_process.pid_term();
            let pid = pid_arc_process.pid_term();

            prop_assert_eq!(result(&arc_process, group_leader, pid), Ok(true.into()));
            prop_assert_eq!(group_leader_0::result(&pid_arc_process), group_leader);

            Ok(())
        }
    );
}

#[test]
fn with_same_group_leader_and_pid_sets_group_leader() {
    run!(
        |arc_process| { (Just(arc_process.clone()), strategy::process()) },
        |(arc_process, group_leader_and_pid_arc_process)| {
            let group_leader = group_leader_and_pid_arc_process.pid_term();
            let pid = group_leader;

            prop_assert_eq!(result(&arc_process, group_leader, pid), Ok(true.into()));
            prop_assert_eq!(
                group_leader_0::result(&group_leader_and_pid_arc_process),
                group_leader
            );

            Ok(())
        }
    );
}
