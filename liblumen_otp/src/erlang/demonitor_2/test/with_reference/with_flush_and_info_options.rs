mod with_monitor;

use super::*;

use crate::erlang::monitor_2;

#[test]
fn without_monitor_returns_false() {
    with_info_option_without_monitor_returns_false(options);
}

fn options(process: &Process) -> Term {
    process
        .list_from_slice(&[Atom::str_to_term("flush"), Atom::str_to_term("info")])
        .unwrap()
}
