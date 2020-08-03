mod with_process;

use super::*;

#[test]
fn without_process_returns_false() {
    with_process_arc(|arc_process| {
        let pid = Pid::next_term();

        assert_eq!(result(&arc_process, pid), Ok(false.into()));
    });
}
