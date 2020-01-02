mod with_local_reference;

use super::*;

#[test]
fn without_local_reference_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_local_reference(arc_process.clone()),
                |timer_reference| {
                    prop_assert_badarg!(
                        native(&arc_process, timer_reference, options(&arc_process)),
                        format!(
                            "timer_reference ({}) is not a local reference",
                            timer_reference
                        )
                    );

                    Ok(())
                },
            )
            .unwrap();
    })
}
