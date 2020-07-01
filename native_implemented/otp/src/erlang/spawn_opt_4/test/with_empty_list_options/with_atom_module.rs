use super::*;

mod with_atom_function;

#[test]
fn without_atom_function_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::atom(),
                strategy::term::is_not_atom(arc_process.clone()),
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
                function
            );

            Ok(())
        },
    );
}
