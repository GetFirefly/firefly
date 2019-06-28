use super::*;

mod with_reference_timer_reference;

#[test]
fn without_reference_timer_reference_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_reference(arc_process.clone()),
                |timer_reference| {
                    let options = Term::EMPTY_LIST;

                    prop_assert_eq!(
                        erlang::cancel_timer_2(timer_reference, options, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_reference_timer_reference_without_list_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_reference(arc_process.clone()),
                    strategy::term::is_not_list(arc_process.clone()),
                ),
                |(timer_reference, options)| {
                    prop_assert_eq!(
                        erlang::cancel_timer_2(timer_reference, options, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
