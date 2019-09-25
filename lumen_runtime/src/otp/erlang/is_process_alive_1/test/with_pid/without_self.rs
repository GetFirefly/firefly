mod with_process;

use super::*;

use liblumen_alloc::erts::term::next_pid;

#[test]
fn without_process_returns_false() {
    with_process_arc(|arc_process| {
        let pid = next_pid();

        assert_eq!(native(&arc_process, pid), Ok(false.into()));
    });
}
