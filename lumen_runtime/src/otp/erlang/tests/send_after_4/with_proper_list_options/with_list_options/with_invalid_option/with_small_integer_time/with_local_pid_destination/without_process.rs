use super::*;

#[test]
fn errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    milliseconds(),
                    strategy::term::heap_fragment_safe(arc_process.clone()),
                ),
                |(milliseconds, message)| {
                    let time = milliseconds.into_process(&arc_process);
                    let destination = process::identifier::local::next();
                    let options = options(&arc_process);

                    prop_assert_eq!(
                        erlang::send_after_4(
                            time,
                            destination,
                            message,
                            options,
                            arc_process.clone()
                        ),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
