mod with_atom;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use crate::erlang::is_boolean_1::result;
use crate::test::strategy;

#[test]
fn without_boolean_returns_false() {
    run!(
        |arc_process| strategy::term::is_not_boolean(arc_process.clone()),
        |term| {
            prop_assert_eq!(result(term), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_boolean_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(&strategy::term::is_boolean(), |term| {
            prop_assert_eq!(result(term), true.into());

            Ok(())
        })
        .unwrap();
}
