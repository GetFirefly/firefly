mod with_atom_module;

use super::*;

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
                result(
                    &arc_process,
                    module,
                    function,
                    arguments,
                    options(&arc_process)
                ),
                module
            );

            Ok(())
        },
    );
}

fn options(process: &Process) -> Term {
    process.list_from_slice(&[atom!("link")]).unwrap()
}
