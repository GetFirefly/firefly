mod with_empty_list_options;
mod with_proper_list_options;

use super::*;

use proptest::arbitrary::any;

#[test]
fn without_proper_list_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::float(arc_process.clone()),
                    strategy::term::is_not_proper_list(arc_process.clone()),
                ),
                |(float, options)| {
                    prop_assert_badarg!(native(&arc_process, float, options), "supported options are compact, {:decimal, 0..253}, or {:scientific, 0..249}");

                    Ok(())
                },
            )
            .unwrap();
    });
}
