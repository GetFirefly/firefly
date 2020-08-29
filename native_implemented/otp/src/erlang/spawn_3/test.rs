mod with_atom_module;

use proptest::strategy::Just;

use crate::erlang::spawn_3;
use crate::test::strategy;

#[test]
fn without_atom_module_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_atom(arc_process.clone()),
                strategy::term::atom(),
                strategy::term::list::proper(arc_process.clone()),
            )
        },
        |(arc_process, module, function, arguments)| {
            prop_assert_is_not_atom!(
                spawn_3::result(&arc_process, module, function, arguments),
                module
            );

            Ok(())
        },
    );
}
