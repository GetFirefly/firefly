mod with_monitor;

use super::*;

use crate::erlang::monitor_2;

// `without_monitor_returns_true`

pub fn options(_: &Process) -> Term {
    Term::NIL
}
