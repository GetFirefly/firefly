use super::*;

use proptest::strategy::Strategy;

mod with_proper_list_options;

#[test]
fn without_list_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term::is_not_list(arc_process.clone()),
                ),
                |(message, options)| {
                    prop_assert_eq!(
                        erlang::send_3(arc_process.pid, message, options, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_not_destination(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process.clone())
        .prop_filter(
            "Destination must not be an atom, pid, or tuple",
            |destination| {
                !(destination.is_atom() || destination.is_pid() || destination.is_tuple())
            },
        )
        .boxed()
}
