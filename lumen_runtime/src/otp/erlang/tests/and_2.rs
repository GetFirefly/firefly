use super::*;

mod with_false_left;
mod with_true_left;

#[test]
fn without_boolean_left_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_boolean(arc_process.clone()),
                    strategy::term::is_boolean(),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::and_2(left, right), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}
