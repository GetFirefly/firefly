use super::*;

mod with_local_reference;

#[test]
fn without_reference_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_reference(arc_process.clone()),
                |timer_reference| {
                    prop_assert_eq!(
                        erlang::cancel_timer_1(timer_reference, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
