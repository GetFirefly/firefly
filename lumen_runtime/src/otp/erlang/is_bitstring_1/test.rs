use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::is_bitstring_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_bitstring_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_bitstring(arc_process.clone()),
                |term| {
                    prop_assert_eq!(native(term), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_bitstring_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_bitstring(arc_process.clone()), |term| {
                prop_assert_eq!(native(term), true.into());

                Ok(())
            })
            .unwrap();
    });
}
