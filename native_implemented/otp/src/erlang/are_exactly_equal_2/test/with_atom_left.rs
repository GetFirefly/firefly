use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_atom_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::atom(),
                strategy::term::is_not_atom(arc_process.clone()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_atom_returns_true() {
    run!(|_| strategy::term::atom(), |operand| {
        prop_assert_eq!(result(operand, operand), true.into());

        Ok(())
    },);
}

#[test]
fn with_different_atom_returns_false() {
    run!(
        |_| {
            (strategy::term::atom(), strategy::term::atom())
                .prop_filter("Atoms must be different", |(left, right)| left != right)
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}
