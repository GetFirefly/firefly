mod with_reference;

use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::demonitor_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

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
