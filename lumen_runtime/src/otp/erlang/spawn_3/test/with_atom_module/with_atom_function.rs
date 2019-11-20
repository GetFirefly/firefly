use super::*;

mod with_empty_list_arguments;
mod with_non_empty_proper_list_arguments;

#[test]
fn without_proper_list_arguments_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term::atom(),
                    strategy::term::is_not_proper_list(arc_process.clone()),
                ),
                |(module, function, arguments)| {
                    prop_assert_eq!(
                        spawn_3::native(&arc_process, module, function, arguments),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
