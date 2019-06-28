use super::*;

mod registered;

#[test]
fn unregistered_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term::atom(), strategy::term(arc_process.clone())),
                |(destination, message)| {
                    prop_assert_eq!(
                        erlang::send_2(destination, message, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
