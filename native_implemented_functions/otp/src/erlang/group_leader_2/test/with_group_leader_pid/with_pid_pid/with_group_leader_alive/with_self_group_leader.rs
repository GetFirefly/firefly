mod with_pid_alive;

use super::*;

#[test]
fn without_pid_alive_returns_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                Just(arc_process.pid_term()),
                strategy::term::pid::local(),
            )
        },
        |(arc_process, group_leader, pid)| {
            prop_assert_badarg!(
                result(&arc_process, group_leader, pid),
                format!("pid ({}) is not alive", pid)
            );

            Ok(())
        }
    );
}
