use crate::otp::erlang::group_leader_0::native;
use crate::otp::erlang::self_0;
use crate::process;

#[test]
fn without_parent_returns_self() {
    let arc_process = process::test_init();

    assert_eq!(native(&arc_process), self_0::native(&arc_process));
}

#[test]
fn with_parent_returns_parent_group_leader() {
    let parent_arc_process = process::test_init();
    let arc_process = process::test(&parent_arc_process);

    assert_eq!(native(&arc_process), self_0::native(&parent_arc_process));
}
