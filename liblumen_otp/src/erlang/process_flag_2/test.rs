mod with_atom_flag;

use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use crate::erlang::process_flag_2::native;
use crate::test::strategy;
use crate::test::with_process;

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
            prop_assert_is_not_atom!(native(&arc_process, flag, value), flag);

            Ok(())
        },
    );
}
