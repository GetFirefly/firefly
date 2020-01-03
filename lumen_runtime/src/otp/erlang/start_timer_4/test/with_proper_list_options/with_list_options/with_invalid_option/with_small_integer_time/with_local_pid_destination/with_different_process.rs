use super::*;

#[test]
fn sends_message_when_timer_expires() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                milliseconds(),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, milliseconds, message)| {
            let time = arc_process.integer(milliseconds).unwrap();

            let destination_arc_process = process::test(&arc_process);
            let destination = destination_arc_process.pid_term();

            let options = options(&arc_process);

            prop_assert_badarg!(
                native(arc_process.clone(), time, destination, message, options),
                "supported option is {:abs, bool}"
            );

            Ok(())
        },
    );
}
