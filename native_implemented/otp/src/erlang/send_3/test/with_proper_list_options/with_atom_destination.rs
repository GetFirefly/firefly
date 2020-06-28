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
            let destination = registered_name();

            prop_assert_badarg!(
                result(&arc_process, destination, message, options),
                format!("name ({}) not registered", destination)
            );

            Ok(())
        },
    );
}
