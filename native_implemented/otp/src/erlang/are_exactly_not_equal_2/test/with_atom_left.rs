use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_atom_returns_true() {
    run!(
        |arc_process| {
            (
                strategy::term::atom(),
                strategy::term::is_not_atom(arc_process.clone()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_atom_returns_false() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(&strategy::term::atom(), |operand| {
            prop_assert_eq!(result(operand, operand), false.into());

            Ok(())
        })
        .unwrap();
}

#[test]
fn with_different_atom_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(strategy::term::atom(), strategy::term::atom())
                .prop_filter("Atoms must be different", |(left, right)| left != right),
            |(left, right)| {
                prop_assert_eq!(result(left, right), true.into());

                Ok(())
            },
        )
        .unwrap();
}
