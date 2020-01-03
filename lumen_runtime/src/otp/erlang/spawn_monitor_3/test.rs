mod with_atom_module;

use std::convert::TryInto;
use std::sync::Arc;

use anyhow::*;

use proptest::strategy::Just;

use liblumen_alloc::erts::process::{Priority, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;
use liblumen_alloc::{atom, atom_from, exit};

use crate::otp::erlang::apply_3;
use crate::otp::erlang::spawn_monitor_3::native;
use crate::process;
use crate::registry::pid_to_process;
use crate::scheduler::Scheduler;
use crate::test::{assert_exits_undef, has_message, run, strategy};

#[test]
fn without_atom_module_errors_badarg() {
    run(
        file!(),
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
