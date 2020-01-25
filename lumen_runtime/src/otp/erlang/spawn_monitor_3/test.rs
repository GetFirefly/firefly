mod with_atom_module;

use std::convert::TryInto;

use anyhow::*;

use proptest::strategy::Just;

use liblumen_alloc::erts::process::{Priority, Status};
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{atom, exit};

use crate::otp::erlang::spawn_monitor_3::native;
use crate::otp::erlang::{self, apply_3};
use crate::process;
use crate::registry::pid_to_process;
use crate::scheduler::Scheduler;
use crate::test::{assert_exits_undef, has_message, strategy};

#[test]
fn without_atom_module_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_atom(arc_process.clone()),
                strategy::term::atom(),
                strategy::term::list::proper(arc_process.clone()),
            )
        },
        |(arc_process, module, function, arguments)| {
            prop_assert_is_not_atom!(native(&arc_process, module, function, arguments), module);

            Ok(())
        },
    );
}
