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
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_map(move |abs| options(abs, &options_arc_process))
                        .boxed(),
                ),
                |(time, message, options)| {
                    let destination = arc_process.pid_term();

                    prop_assert_eq!(
                        native(arc_process.clone(), time, destination, message, options,),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn options(abs: Term, process: &Process) -> Term {
    process
        .cons(
            process
                .tuple_from_slice(&[Atom::str_to_term("abs"), abs])
                .unwrap(),
            Term::NIL,
        )
        .unwrap()
}
