mod with_atom_name;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::{Atom, Encoded, Pid};

use crate::otp::erlang;
use crate::otp::erlang::register_2::native;
use crate::scheduler::with_process_arc;
use crate::test::{registered_name, run, strategy};
use crate::{process, registry};

#[test]
fn without_atom_name_errors_badarg() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_atom(arc_process.clone()),
                strategy::term::pid_or_port(arc_process.clone()),
            )
        },
        |(arc_process, name, pid_or_port)| {
            prop_assert_is_not_atom!(native(arc_process.clone(), name, pid_or_port), name);

            Ok(())
        },
    );
}
