use super::*;

use proptest::strategy::Strategy;

mod with_atom;

#[test]
fn without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        let options_arc_process = arc_process.clone();

        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_non_negative_integer(arc_process.clone()),
                    strategy::term::heap_fragment_safe(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_map(move |abs| options(abs, &options_arc_process))
                        .boxed(),
                ),
                |(time, message, options)| {
                    let destination = arc_process.pid;

                    prop_assert_eq!(
                        erlang::start_timer_4(
                            time,
                            destination,
                            message,
                            options,
                            arc_process.clone()
                        ),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn options(abs: Term, process: &Process) -> Term {
    Term::cons(
        Term::slice_to_tuple(
            &[Term::str_to_atom("abs", DoNotCare).unwrap(), abs],
            process,
        ),
        Term::EMPTY_LIST,
        process,
    )
}
