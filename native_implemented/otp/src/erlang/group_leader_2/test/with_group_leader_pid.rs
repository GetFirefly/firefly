mod with_pid_pid;

use super::*;

#[test]
fn without_pid_pid_returns_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                prop_oneof![Just(arc_process.clone()), strategy::process()],
                strategy::term::is_not_pid(arc_process),
            )
        },
        |(arc_process, group_leader_arc_process, pid)| {
            let group_leader = group_leader_arc_process.pid_term();

            prop_assert_is_not_local_pid!(result(&arc_process, group_leader, pid), pid);

            Ok(())
        }
    );
}
