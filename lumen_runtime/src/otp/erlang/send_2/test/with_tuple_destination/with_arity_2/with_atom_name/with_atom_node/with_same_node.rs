use super::*;

mod registered;

#[test]
fn unregistered_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::atom(),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, name, message)| {
            let destination = arc_process
                .tuple_from_slice(&[name, erlang::node_0::native()])
                .unwrap();

            prop_assert_badarg!(
                native(&arc_process, destination, message),
                format!("name ({}) not registered", name)
            );

            Ok(())
        },
    );
}
