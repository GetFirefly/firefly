use super::*;

#[test]
fn errors_badarg() {
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
            let destination = registered_name();

            prop_assert_eq!(
                erlang::register_2::native(
                    arc_process.clone(),
                    destination,
                    destination_arc_process.pid_term(),
                ),
                Ok(true.into())
            );

            let options = options(&arc_process);

            prop_assert_badarg!(
                native(arc_process.clone(), time, destination, message, options),
                "supported option is {:abs, bool}"
            );

            Ok(())
        },
    );
}
