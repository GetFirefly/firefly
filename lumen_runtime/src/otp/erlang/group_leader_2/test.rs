mod with_group_leader_pid;

use proptest::strategy::Just;
use proptest::{prop_assert_eq, prop_oneof};

use crate::otp::erlang::group_leader_0;
use crate::otp::erlang::group_leader_2::native;
use crate::scheduler::with_process;
use crate::test::strategy;

#[test]
fn without_group_leader_pid_returns_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_pid(arc_process),
                strategy::term::pid::local(),
            )
        },
        |(arc_process, group_leader, pid)| {
            prop_assert_is_not_local_pid!(native(&arc_process, group_leader, pid), group_leader);

            Ok(())
        }
    );
}
