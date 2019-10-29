mod with_atom_module;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::exception::{RuntimeException, Exception};
use liblumen_alloc::erts::process::{Priority, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;
use liblumen_alloc::{badarg, badarith, exit, undef, atom};

use crate::otp::erlang::apply_3;
use crate::otp::erlang::spawn_monitor_3::native;
use crate::process;
use crate::registry::pid_to_process;
use crate::scheduler::{with_process_arc, Scheduler};
use crate::test::has_message;
use crate::test::strategy;

#[test]
fn without_atom_module_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term::atom(),
                    strategy::term::list::proper(arc_process.clone()),
                ),
                |(module, function, arguments)| {
                    prop_assert_eq!(
                        native(&arc_process, module, function, arguments),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
