use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::throw;

use crate::otp::erlang::throw_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn throws_reason() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |reason| {
                prop_assert_eq!(
                    native(&arc_process, reason),
                    Err(throw!(&arc_process, reason).into())
                );

                Ok(())
            })
            .unwrap();
    });
}
