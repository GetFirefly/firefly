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
                    prop_assert_is_not_arity!(native(function, arity), arity);

                    Ok(())
                },
            )
            .unwrap();
    });
}
