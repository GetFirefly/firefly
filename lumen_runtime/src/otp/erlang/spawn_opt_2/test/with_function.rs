use super::*;

mod with_empty_list_options;
mod with_link_and_monitor_in_options_list;
mod with_link_in_options_list;
mod with_monitor_in_options_list;

#[test]
fn without_proper_list_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_function(arc_process.clone()),
                    strategy::term::is_not_proper_list(arc_process.clone()),
                ),
                |(function, options)| {
                    prop_assert_eq!(
                        native(&arc_process, function, options),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
