use super::*;

#[test]
fn with_different_process_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                milliseconds(),
                strategy::term(arc_process.clone()),
                abs_value(arc_process.clone()),
            )
        },
        |(arc_process, milliseconds, message, abs_value)| {
            let time = arc_process.integer(milliseconds).unwrap();

            let destination_arc_process = test::process::child(&arc_process);
            let destination = destination_arc_process.pid_term();

            let options = options(abs_value, &arc_process);

            prop_assert_is_not_boolean!(
                result(arc_process.clone(), time, destination, message, options),
                "abs value",
                abs_value
            );

            Ok(())
        },
    );
}

#[test]
fn with_same_process_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                milliseconds(),
                strategy::term(arc_process.clone()),
                abs_value(arc_process),
            )
        },
        |(arc_process, milliseconds, message, abs_value)| {
            let time = arc_process.integer(milliseconds).unwrap();
            let destination = arc_process.pid_term();
            let options = options(abs_value, &arc_process);

            prop_assert_is_not_boolean!(
                result(arc_process.clone(), time, destination, message, options),
                "abs value",
                abs_value
            );

            Ok(())
        },
    );
}

#[test]
fn without_process_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                milliseconds(),
                strategy::term(arc_process.clone()),
                abs_value(arc_process.clone()),
            )
        },
        |(arc_process, milliseconds, message, abs_value)| {
            let time = arc_process.integer(milliseconds).unwrap();
            let destination = Pid::next_term();
            let options = options(abs_value, &arc_process);

            prop_assert_is_not_boolean!(
                result(arc_process.clone(), time, destination, message, options),
                "abs value",
                abs_value
            );

            Ok(())
        },
    );
}
