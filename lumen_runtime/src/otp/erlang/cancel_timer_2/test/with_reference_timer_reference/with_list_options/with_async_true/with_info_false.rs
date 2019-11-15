use super::*;

mod with_local_reference;

#[test]
fn without_local_reference_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_local_reference(arc_process.clone()),
                |timer_reference| {
                    prop_assert_eq!(
                        native(&arc_process, timer_reference, options(&arc_process)),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    })
}

fn options(process: &Process) -> Term {
    process
        .cons(info_option(false, process), super::options(process))
        .unwrap()
}
