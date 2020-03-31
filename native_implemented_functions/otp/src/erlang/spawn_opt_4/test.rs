mod with_empty_list_options;
mod with_link_in_options_list;

use std::convert::TryInto;
use std::sync::Arc;

use anyhow::*;

use proptest::strategy::Just;

use liblumen_alloc::erts::process::{Priority, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{atom, atom_from, exit, ModuleFunctionArity, Process};

use crate::runtime::registry::pid_to_process;
use crate::runtime::scheduler;

use crate::erlang;
use crate::erlang::apply_3;
use crate::erlang::spawn_opt_4::native;
use crate::test;
use crate::test::{assert_exits_badarith, assert_exits_undef, strategy};

#[test]
fn without_proper_list_options_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::atom(),
                strategy::term::atom(),
                strategy::term::list::proper(arc_process.clone()),
                strategy::term::is_not_proper_list(arc_process.clone()),
            )
        },
        |(arc_process, module, function, arguments, options)| {
            prop_assert_badarg!(
                native(&arc_process, module, function, arguments, options),
                SUPPORTED_OPTIONS
            );

            Ok(())
        },
    );
}

const SUPPORTED_OPTIONS: &str = "supported options are :link, :monitor, \
                                 {:fullsweep_after, generational_collections :: pos_integer()}, \
                                 {:max_heap_size, words :: pos_integer()}, \
                                 {:message_queue_data, :off_heap | :on_heap}, \
                                 {:min_bin_vheap_size, words :: pos_integer()}, \
                                 {:min_heap_size, words :: pos_integer()}, and \
                                 {:priority, level :: :low | :normal | :high | :max}";
