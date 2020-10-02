use super::*;

// `without_exiting_returns_true` in integration tests

#[test]
fn with_exiting_returns_false() {
    with_process_arc(|arc_process| {
        arc_process.exit_normal();

        assert!(arc_process.is_exiting());
        assert_eq!(
            result(&arc_process, arc_process.pid_term()),
            Ok(false.into())
        );
    });
}
