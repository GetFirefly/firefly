use super::*;

mod registered;

#[test]
fn unregistered_sends_nothing_when_timer_expires() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    milliseconds(),
                    strategy::term::heap_fragment_safe(arc_process.clone()),
                ),
                |(milliseconds, message)| {
                    let time = arc_process.integer(milliseconds);
                    let destination = registered_name();
                    let options = options(&arc_process);

                    let result = erlang::send_after_4(
                        time,
                        destination,
                        message,
                        options,
                        arc_process.clone(),
                    );

                    prop_assert!(
                        result.is_ok(),
                        "Timer reference not returned.  Got {:?}",
                        result
                    );

                    let timer_reference = result.unwrap();

                    prop_assert!(timer_reference.is_local_reference());
                    prop_assert!(!has_message(&arc_process, message));

                    thread::sleep(Duration::from_millis(milliseconds + 1));
                    timer::timeout().unwrap();

                    prop_assert!(!has_message(&arc_process, message));

                    Ok(())
                },
            )
            .unwrap();
    });
}
