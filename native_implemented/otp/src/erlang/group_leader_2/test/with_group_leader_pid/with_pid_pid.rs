mod with_group_leader_alive;

use super::*;

#[test]
fn without_group_leader_alive_returns_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process),
                strategy::term::pid::local(),
                strategy::term::pid::local(),
            )
        },
        |(arc_process, group_leader, pid)| {
            prop_assert_badarg!(
                result(&arc_process, group_leader, pid),
                format!("group_leader ({}) is not alive", group_leader)
            );

            Ok(())
        }
    );
}
