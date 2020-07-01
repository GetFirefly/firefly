mod with_reference;

use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::demonitor_1::result;
use crate::erlang::monitor_2;
use crate::runtime::scheduler::{self, SchedulerDependentAlloc};
use crate::test::{
    self, exit_when_run, has_message, monitor_count, monitored_count, strategy, with_process_arc,
};

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
            prop_assert_is_not_local_reference!(result(&arc_process, reference), reference);

            Ok(())
        },
    );
}
