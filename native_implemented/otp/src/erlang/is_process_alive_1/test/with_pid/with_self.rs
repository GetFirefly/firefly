use super::*;

#[test]
fn without_exiting_returns_true() {
    with_process_arc(|arc_process| {
        assert!(!arc_process.is_exiting());
        assert_eq!(
            result(&arc_process, arc_process.pid_term()),
            Ok(true.into())
        );
    });
}

#[test]
fn with_exiting_returns_false() {
    with_process_arc(|arc_process| {
        arc_process.exit_normal(anyhow!("Test").into());

        assert!(arc_process.is_exiting());
        assert_eq!(
            result(&arc_process, arc_process.pid_term()),
            Ok(false.into())
        );
    });
}
