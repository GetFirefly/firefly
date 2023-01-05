mod with_atom_flag;

use proptest::strategy::{BoxedStrategy, Just};

use firefly_rt::term::{atoms, Atom, Term};

use crate::erlang::process_flag_2::result;
use crate::test::*;

#[test]
fn without_atom_flag_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_atom(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, flag, value)| {
            prop_assert_is_not_atom!(result(&arc_process, flag, value), flag);

            Ok(())
        },
    );
}
