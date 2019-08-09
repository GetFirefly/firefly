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

                    prop_assert_eq!(
                        erlang::send_after_4(
                            time,
                            destination,
                            message,
                            options,
                            arc_process.clone()
                        ),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
