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
        |(arc_process, destination, message)| {
            prop_assert_badarg!(
                result(&arc_process, destination, message),
                format!("name ({}) not registered", destination)
            );

            Ok(())
        },
    );
}
