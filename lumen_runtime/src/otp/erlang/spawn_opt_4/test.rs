mod with_empty_list_options;
mod with_link_in_options_list;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::exception::RuntimeException;
use liblumen_alloc::erts::process::{Priority, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{badarg, badarith, exit, undef, atom, ModuleFunctionArity, Process};

use crate::otp::erlang::apply_3;
use crate::otp::erlang::spawn_opt_4::native;
use crate::process;
use crate::registry::pid_to_process;
use crate::scheduler::{with_process_arc, Scheduler};
use crate::test::strategy;

#[test]
fn without_proper_list_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term::atom(),
                    strategy::term::list::proper(arc_process.clone()),
                    strategy::term::is_not_proper_list(arc_process.clone()),
                ),
                |(module, function, arguments, options)| {
                    prop_assert_eq!(
                        native(&arc_process, module, function, arguments, options),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
