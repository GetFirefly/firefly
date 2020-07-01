use super::*;

mod registered;

#[test]
fn unregistered_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                valid_options(arc_process.clone()),
            )
        },
        |(arc_process, message, options)| {
            let name = registered_name();
            let destination = arc_process
                .tuple_from_slice(&[name, erlang::node_0::result()])
                .unwrap();

            prop_assert_badarg!(
                result(&arc_process, destination, message, options),
                format!("name ({}) not registered", name)
            );

            Ok(())
        },
    );
}
