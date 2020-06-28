mod with_atom_name;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::{Atom, Encoded, Pid};

use crate::runtime::registry;

use crate::erlang;
use crate::erlang::register_2::result;
use crate::test;
use crate::test::{registered_name, strategy, with_process_arc};

#[test]
fn without_atom_name_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_atom(arc_process.clone()),
                strategy::term::pid_or_port(arc_process.clone()),
            )
        },
        |(arc_process, name, pid_or_port)| {
            prop_assert_is_not_atom!(result(arc_process.clone(), name, pid_or_port), name);

            Ok(())
        },
    );
}
