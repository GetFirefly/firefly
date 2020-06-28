use crate::erlang::group_leader_0::result;
use crate::erlang::self_0;
use crate::test;

#[test]
fn without_parent_returns_self() {
    let arc_process = test::process::init();

    assert_eq!(result(&arc_process), self_0::result(&arc_process));
}

#[test]
fn with_parent_returns_parent_group_leader() {
    let parent_arc_process = test::process::init();
    let arc_process = test::process::child(&parent_arc_process);

    assert_eq!(result(&arc_process), self_0::result(&parent_arc_process));
}
