use anyhow::*;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::error;

use crate::otp::erlang::error_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn errors_with_reason() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |reason| {
                prop_assert_eq!(
                    native(reason),
                    Err(error!(reason, anyhow!("test").into()).into())
                );

                Ok(())
            })
            .unwrap();
    });
}
