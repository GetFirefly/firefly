use super::*;

#[test]
fn with_self_pid_sets_group_leader() {
    with_process(|process| {
        let group_leader = process.pid_term();
        let pid = process.pid_term();

        assert_eq!(native(process, group_leader, pid), Ok(true.into()));
        assert_eq!(group_leader_0::native(process), group_leader);
    });
}

#[test]
fn without_self_pid_sets_group_leader() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                Just(arc_process.pid_term()),
                strategy::process(),
            )
        },
        |(arc_process, group_leader, pid_arc_process)| {
            let pid = pid_arc_process.pid_term();

            prop_assert_eq!(native(&arc_process, group_leader, pid), Ok(true.into()));
            prop_assert_eq!(group_leader_0::native(&pid_arc_process), group_leader);

            Ok(())
        }
    );
}
