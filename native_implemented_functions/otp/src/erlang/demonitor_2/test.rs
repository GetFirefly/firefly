mod with_reference;

use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::atom;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::scheduler::{self, SchedulerDependentAlloc};

use crate::erlang::demonitor_2::native;
use crate::erlang::{exit_1, monitor_2};
use crate::test::{self, has_message, monitor_count, monitored_count, strategy, with_process_arc};

#[test]
fn without_reference_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_reference(arc_process.clone()),
            )
        },
        |(arc_process, reference)| {
            let options = Term::NIL;

            prop_assert_is_not_local_reference!(
                native(&arc_process, reference, options),
                reference
            );

            Ok(())
        },
    );
}
