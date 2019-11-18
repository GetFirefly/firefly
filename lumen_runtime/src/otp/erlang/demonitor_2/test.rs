mod with_reference;

use proptest::prop_assert_eq;
use proptest::strategy::{BoxedStrategy, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::demonitor_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_reference_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_reference(arc_process.clone()),
                |reference| {
                    let options = Term::NIL;

                    prop_assert_eq!(
                        native(&arc_process, reference, options),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
