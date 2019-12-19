use super::*;

mod with_atom;

#[test]
fn without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_non_negative_integer(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(time, message, abs_value)| {
                    let options = options(abs_value, &arc_process);
                    let destination = arc_process.pid_term();

                    prop_assert_is_not_boolean!(
                        native(arc_process.clone(), time, destination, message, options),
                        "abs value",
                        abs_value
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
