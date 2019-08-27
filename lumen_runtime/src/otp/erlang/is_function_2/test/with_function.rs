mod with_non_negative_arity;

use super::*;

#[test]
fn without_non_negative_arity_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_function(arc_process.clone()),
                    strategy::term::integer::negative(arc_process.clone()),
                ),
                |(function, arity)| {
                    prop_assert_eq!(native(function, arity), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}
