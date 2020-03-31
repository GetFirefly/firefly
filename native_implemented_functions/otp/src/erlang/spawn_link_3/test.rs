mod with_atom_module;

use std::convert::TryInto;
use std::sync::Arc;

use anyhow::*;

use proptest::strategy::Just;

use liblumen_alloc::erts::process::{Priority, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;
use liblumen_alloc::{atom, atom_from, exit};

use lumen_rt_full::registry::pid_to_process;
use lumen_rt_full::scheduler::Scheduler;

use crate::erlang::apply_3;
use crate::erlang::spawn_link_3::native;
use crate::test::{assert_exits_badarith, assert_exits_undef, strategy};
use crate::{erlang, test};

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
