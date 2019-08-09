use super::*;

mod with_empty_list;
mod with_non_empty_proper_list;

#[test]
fn without_list_left_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_list(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::concatenate_2(left, right, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_improper_list_left_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::list::improper(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::concatenate_2(left, right, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
