mod with_other;
mod with_self;

use super::*;

use liblumen_alloc::erts::term::next_pid;

#[test]
fn without_process_returns_undefined() {
    with_process_arc(|arc_process| {
        let pid = next_pid();

        assert_eq!(
            native(&arc_process, pid, item()),
            Ok(atom_unchecked("undefined"))
        );
    });
}

fn item() -> Term {
    atom_unchecked("registered_name")
}
