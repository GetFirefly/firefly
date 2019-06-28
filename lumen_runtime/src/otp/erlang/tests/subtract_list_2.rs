use super::*;

mod with_proper_list_minuend;

#[test]
fn without_proper_list_minuend_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_proper_list(arc_process.clone()),
                    strategy::term::is_list(arc_process.clone()),
                ),
                |(minuend, subtrahend)| {
                    prop_assert_eq!(
                        erlang::subtract_list_2(minuend, subtrahend, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
