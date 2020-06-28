mod with_other;
mod with_self;

use super::*;

#[test]
fn without_process_returns_undefined() {
    with_process_arc(|arc_process| {
        let pid = Pid::next_term();

        assert_eq!(
            result(&arc_process, pid, item()),
            Ok(Atom::str_to_term("undefined"))
        );
    });
}

fn item() -> Term {
    Atom::str_to_term("registered_name")
}
