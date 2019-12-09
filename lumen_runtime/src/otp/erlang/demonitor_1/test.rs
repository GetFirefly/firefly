mod with_reference;

use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::Term;

use crate::otp::erlang::demonitor_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_reference_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_reference(arc_process.clone()),
                |reference| {
                    prop_assert_badarg!(
                        native(&arc_process, reference),
                        format!("reference ({}) must be a reference", reference)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
