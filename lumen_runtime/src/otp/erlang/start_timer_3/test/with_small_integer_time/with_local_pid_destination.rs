use super::*;

#[test]
fn with_different_process_sends_message_when_timer_expires() {
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

            let result =
                erlang::start_timer_3::native(arc_process.clone(), time, destination, message);

            prop_assert!(
                result.is_ok(),
                "Timer reference not returned.  Got {:?}",
                result
            );

            let timer_reference = result.unwrap();

            prop_assert!(timer_reference.is_boxed_local_reference());

            let timeout_message = timeout_message(timer_reference, message, &arc_process);

            prop_assert!(!has_message(&destination_arc_process, timeout_message));

            thread::sleep(Duration::from_millis(milliseconds + 1));

            timer::timeout();

            prop_assert!(has_message(&destination_arc_process, timeout_message));

            Ok(())
        },
    );
}

#[test]
fn with_same_process_sends_message_when_timer_expires() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                milliseconds(),
                strategy::term(arc_process),
            )
        },
        |(arc_process, milliseconds, message)| {
            let time = arc_process.integer(milliseconds).unwrap();
            let destination = arc_process.pid_term();

            let result =
                erlang::start_timer_3::native(arc_process.clone(), time, destination, message);

            prop_assert!(
                result.is_ok(),
                "Timer reference not returned.  Got {:?}",
                result
            );

            let timer_reference = result.unwrap();

            prop_assert!(timer_reference.is_boxed_local_reference());

            let timeout_message = timeout_message(timer_reference, message, &arc_process);

            prop_assert!(!has_message(&arc_process, timeout_message));

            thread::sleep(Duration::from_millis(milliseconds + 1));
            timer::timeout();

            prop_assert!(has_message(&arc_process, timeout_message));

            Ok(())
        },
    );
}

#[test]
fn without_process_sends_nothing_when_timer_expires() {
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
            let destination = Pid::next_term();

            let result =
                erlang::start_timer_3::native(arc_process.clone(), time, destination, message);

            prop_assert!(
                result.is_ok(),
                "Timer reference not returned.  Got {:?}",
                result
            );

            let timer_reference = result.unwrap();

            prop_assert!(timer_reference.is_boxed_local_reference());

            let timeout_message = arc_process
                .tuple_from_slice(&[Atom::str_to_term("timeout"), timer_reference, message])
                .unwrap();

            thread::sleep(Duration::from_millis(milliseconds + 1));
            timer::timeout();

            // does not send to original process either
            prop_assert!(!has_message(&arc_process, timeout_message));

            Ok(())
        },
    );
}
