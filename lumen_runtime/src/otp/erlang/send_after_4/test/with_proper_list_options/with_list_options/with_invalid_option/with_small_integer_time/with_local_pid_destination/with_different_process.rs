use super::*;

#[test]
fn sends_message_when_timer_expires() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(milliseconds(), strategy::term(arc_process.clone())),
                |(milliseconds, message)| {
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
            )
            .unwrap();
    });
}
