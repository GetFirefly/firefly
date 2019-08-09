use super::*;

mod with_proper_list_options;

#[test]
fn without_list_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term::is_not_list(arc_process.clone()),
                ),
                |(message, options)| {
                    prop_assert_eq!(
                        erlang::send_3(arc_process.pid_term(), message, options, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
