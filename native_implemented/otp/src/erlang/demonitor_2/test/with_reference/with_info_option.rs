mod with_monitor;

use super::*;

use crate::erlang::monitor_2;

// `without_monitor_returns_false` in integration tests

fn options(process: &Process) -> Term {
    process
        .list_from_slice(&[Atom::str_to_term("info")])
        .unwrap()
}
