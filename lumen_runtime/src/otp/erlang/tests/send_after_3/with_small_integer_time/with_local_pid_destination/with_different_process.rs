use super::*;

use std::thread;
use std::time::Duration;

#[test]
fn sends_message_when_timer_expires() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    milliseconds(),
                    strategy::term::heap_fragment_safe(arc_process.clone()),
                ),
                |(milliseconds, message)| {
                    let time = milliseconds.into_process(&arc_process);

                    let destination_arc_process = process::local::test(&arc_process);
                    let destination = destination_arc_process.pid;

                    let result =
                        erlang::send_after_3(time, destination, message, arc_process.clone());

                    assert!(
                        result.is_ok(),
                        "Timer reference not returned.  Got {:?}",
                        result
                    );

                    let timer_reference = result.unwrap();

                    assert_eq!(timer_reference.tag(), Boxed);

                    let unboxed_timer_reference: &Term = timer_reference.unbox_reference();

                    assert_eq!(unboxed_timer_reference.tag(), LocalReference);

                    assert!(!has_message(&destination_arc_process, message));

                    thread::sleep(Duration::from_millis(milliseconds + 1));

                    timer::timeout();

                    assert!(has_message(&destination_arc_process, message));

                    Ok(())
                },
            )
            .unwrap();
    });
}
