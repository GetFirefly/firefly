use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use crate::erlang::is_function_1::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_function_returns_false() {
    run!(
        |arc_process| strategy::term::is_not_function(arc_process.clone()),
        |term| {
            prop_assert_eq!(result(term), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_function_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_function(arc_process.clone()), |term| {
                prop_assert_eq!(result(term), true.into());

                Ok(())
            })
            .unwrap();
    });
}
