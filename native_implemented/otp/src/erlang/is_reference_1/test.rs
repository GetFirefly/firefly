use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use crate::erlang::is_reference_1::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_reference_returns_false() {
    run!(
        |arc_process| strategy::term::is_not_reference(arc_process.clone()),
        |term| {
            prop_assert_eq!(result(term), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_tuple_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_reference(arc_process.clone()), |term| {
                prop_assert_eq!(result(term), true.into());

                Ok(())
            })
            .unwrap();
    });
}
