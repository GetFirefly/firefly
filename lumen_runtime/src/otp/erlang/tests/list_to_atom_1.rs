use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(erlang::list_to_atom_1(list), Err(badarg!()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_returns_empty_atom() {
    assert_eq!(
        erlang::list_to_atom_1(Term::EMPTY_LIST),
        Ok(Term::str_to_atom("", DoNotCare).unwrap())
    );
}

#[test]
fn with_improper_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::list::improper(arc_process.clone()),
                |list| {
                    prop_assert_eq!(erlang::list_to_atom_1(list), Err(badarg!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_non_empty_proper_list_returns_atom() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &any::<String>().prop_map(|string| {
                    let codepoint_terms: Vec<Term> = string
                        .chars()
                        .map(|c| c.into_process(&arc_process))
                        .collect();
                    let list = Term::slice_to_list(&codepoint_terms, &arc_process);

                    (list, string)
                }),
                |(list, string)| {
                    let atom = Term::str_to_atom(&string, DoNotCare).unwrap();

                    prop_assert_eq!(erlang::list_to_atom_1(list), Ok(atom));

                    Ok(())
                },
            )
            .unwrap();
    });
}
