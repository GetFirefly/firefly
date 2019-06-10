use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

mod with_reference;

#[test]
fn without_reference_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_reference(arc_process.clone()),
                    options(arc_process.clone()),
                ),
                |(timer_reference, options)| {
                    prop_assert_eq!(
                        erlang::read_timer_2(timer_reference, options, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_reference_without_list_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_reference(arc_process.clone()),
                    strategy::term::is_not_list(arc_process.clone()),
                ),
                |(timer_reference, options)| {
                    prop_assert_eq!(
                        erlang::read_timer_2(timer_reference, options, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn async_option(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term::is_boolean()
        .prop_map(move |async_value| {
            Term::slice_to_tuple(
                &[Term::str_to_atom("async", DoNotCare).unwrap(), async_value],
                &arc_process,
            )
        })
        .boxed()
}

fn options(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        Just(Term::EMPTY_LIST),
        async_option(arc_process.clone())
            .prop_map(move |async_option| { Term::slice_to_list(&[async_option], &arc_process) })
    ]
    .boxed()
}
