use super::*;

mod with_atom_name;

#[test]
fn without_atom_name_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_atom(arc_process.clone()),
                strategy::term(arc_process.clone()),
                valid_options(arc_process.clone()),
            )
        },
        |(arc_process, name, message, options)| {
            let destination = arc_process
                .tuple_from_slice(&[name, erlang::node_0::result()])
                .unwrap();

            prop_assert_badarg!(
                        result(&arc_process, destination, message, options),
                        format!("registered_name ({}) in {{registered_name, node}} ({}) destination is not an atom", name, destination)
                    );

            Ok(())
        },
    );
}
