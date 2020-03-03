mod with_reference;

use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::Term;

use crate::otp::erlang::demonitor_1::native;
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
            prop_assert_is_not_local_reference!(native(&arc_process, reference), reference);

            Ok(())
        },
    );
}
