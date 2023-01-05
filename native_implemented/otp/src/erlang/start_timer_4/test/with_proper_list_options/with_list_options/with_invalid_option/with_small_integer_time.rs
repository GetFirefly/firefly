use super::*;

mod with_atom_destination;
mod with_local_pid_destination;

#[test]
fn without_atom_pid_or_tuple_destination_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                milliseconds(),
                strategy::term::is_not_send_after_destination(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, milliseconds, destination, message)| {
            let time = arc_process.integer(milliseconds).unwrap();
            let options = options(&arc_process);

            prop_assert_badarg!(
                result(arc_process.clone(), time, destination, message, options),
                "supported option is {:abs, bool}"
            );

            Ok(())
        },
    );
}
