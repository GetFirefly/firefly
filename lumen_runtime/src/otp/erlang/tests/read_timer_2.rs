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
                        Err(badarg!().into())
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
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn async_option(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    strategy::term::is_boolean()
        .prop_map(move |async_value| {
            arc_process
                .tuple_from_slice(&[atom_unchecked("async"), async_value])
                .unwrap()
        })
        .boxed()
}

fn options(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    prop_oneof![
        Just(Term::NIL),
        async_option(arc_process.clone()).prop_map(move |async_option| {
            arc_process.list_from_slice(&[async_option]).unwrap()
        })
    ]
    .boxed()
}
