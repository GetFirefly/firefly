use super::*;

#[test]
fn with_different_process_sends_message_when_timer_expires() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(milliseconds(), strategy::process()).prop_flat_map(|(milliseconds, arc_process)| {
                (
                    Just(milliseconds),
                    Just(arc_process.clone()),
                    strategy::term::heap_fragment_safe(arc_process),
                )
            }),
            |(milliseconds, arc_process, message)| {
                let time = arc_process.integer(milliseconds).unwrap();

                let destination_arc_process = process::test(&arc_process);
                let destination = destination_arc_process.pid_term();

                let options = options(&arc_process);

                let result =
                    erlang::start_timer_4(time, destination, message, options, arc_process.clone());

                prop_assert!(
                    result.is_ok(),
                    "Timer reference not returned.  Got {:?}",
                    result
                );

                let timer_reference = result.unwrap();

                prop_assert!(timer_reference.is_local_reference());

                let timeout_message = arc_process
                    .tuple_from_slice(&[atom_unchecked("timeout"), timer_reference, message])
                    .unwrap();

                prop_assert!(!has_message(&destination_arc_process, timeout_message));

                // No sleeping is necessary because timeout is in the past and so the timer will
                // timeout at once

                timer::timeout().unwrap();

                prop_assert!(has_message(&destination_arc_process, timeout_message));

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_same_process_sends_message_when_timer_expires() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(milliseconds(), strategy::process()).prop_flat_map(|(milliseconds, arc_process)| {
                (
                    Just(milliseconds),
                    Just(arc_process.clone()),
                    strategy::term::heap_fragment_safe(arc_process),
                )
            }),
            |(milliseconds, arc_process, message)| {
                let time = arc_process.integer(milliseconds).unwrap();
                let destination = arc_process.pid_term();
                let options = options(&arc_process);

                let result =
                    erlang::start_timer_4(time, destination, message, options, arc_process.clone());

                prop_assert!(
                    result.is_ok(),
                    "Timer reference not returned.  Got {:?}",
                    result
                );

                let timer_reference = result.unwrap();

                prop_assert!(timer_reference.is_local_reference());

                let timeout_message = arc_process
                    .tuple_from_slice(&[atom_unchecked("timeout"), timer_reference, message])
                    .unwrap();

                prop_assert!(!has_message(&arc_process, timeout_message));

                // No sleeping is necessary because timeout is in the past and so the timer will
                // timeout at once

                timer::timeout().unwrap();

                prop_assert!(has_message(&arc_process, timeout_message));

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn without_process_sends_nothing_when_timer_expires() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(milliseconds(), strategy::process()).prop_flat_map(|(milliseconds, arc_process)| {
                (
                    Just(milliseconds),
                    Just(arc_process.clone()),
                    strategy::term::heap_fragment_safe(arc_process),
                )
            }),
            |(milliseconds, arc_process, message)| {
                let time = arc_process.integer(milliseconds).unwrap();
                let destination = next_pid();
                let options = options(&arc_process);

                let result =
                    erlang::start_timer_4(time, destination, message, options, arc_process.clone());

                prop_assert!(
                    result.is_ok(),
                    "Timer reference not returned.  Got {:?}",
                    result
                );

                let timer_reference = result.unwrap();

                prop_assert!(timer_reference.is_local_reference());

                let timeout_message = arc_process
                    .tuple_from_slice(&[atom_unchecked("timeout"), timer_reference, message])
                    .unwrap();

                prop_assert!(!has_message(&arc_process, timeout_message));

                // No sleeping is necessary because timeout is in the past and so the timer will
                // timeout at once

                timer::timeout().unwrap();

                prop_assert!(!has_message(&arc_process, timeout_message));

                Ok(())
            },
        )
        .unwrap();
}
