mod with_atom_class;

use proptest::prop_assert_eq;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::raise_3::result;
use crate::test::strategy;

#[test]
fn without_atom_class_errors_badarg() {
    run!(
        |arc_process| {
            (
                strategy::term::is_not_atom(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term::list::proper(arc_process.clone()),
            )
        },
        |(class, reason, stacktrace)| {
            prop_assert_badarg!(
                result(class, reason, stacktrace),
                format!("class ({}) is not an atom", class)
            );

            Ok(())
        },
    );
}
