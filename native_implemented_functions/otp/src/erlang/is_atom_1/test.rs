use proptest::prop_assert_eq;

use crate::erlang::is_atom_1::result;
use crate::test::strategy;

#[test]
fn without_atom_returns_false() {
    run!(
        |arc_process| strategy::term::is_not_atom(arc_process.clone()),
        |term| {
            prop_assert_eq!(result(term), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_atom_returns_true() {
    run!(|_| strategy::term::atom(), |term| {
        prop_assert_eq!(result(term), true.into());

        Ok(())
    },);
}
