use super::*;

#[test]
fn with_different_node_returns_noconnect() {
    run!(
        |arc_process| { (Just(arc_process.clone()), strategy::term(arc_process)) },
        |(arc_process, message)| {
            let name = registered_name();

            prop_assert_eq!(
                erlang::register_2::result(arc_process.clone(), name, arc_process.pid_term()),
                Ok(true.into())
            );

            let destination =
                arc_process.tuple_term_from_term_slice(&[name, Atom::str_to_term("node@example.com")]);
            let options = options(&arc_process);

            prop_assert_eq!(
                result(&arc_process, destination, message, options),
                Ok(Atom::str_to_term("noconnect"))
            );

            Ok(())
        },
    );
}
