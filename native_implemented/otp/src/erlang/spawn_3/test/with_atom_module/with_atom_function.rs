use super::*;

// `with_empty_list_arguments` in integration tests
// `with_non_empty_proper_list_arguments` in integration tests

#[test]
fn without_atom_function_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::atom(),
                strategy::term::atom(),
                strategy::term::is_not_proper_list(arc_process.clone()),
            )
        },
        |(arc_process, module, function, arguments)| {
            prop_assert_badarg!(
                spawn_3::result(&arc_process, module, function, arguments),
                format!("arguments ({}) is not a proper list", arguments)
            );

            Ok(())
        },
    );
}
