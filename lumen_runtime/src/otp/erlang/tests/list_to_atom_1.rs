use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(erlang::list_to_atom_1(list), Err(badarg!().into()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_returns_empty_atom() {
    assert_eq!(erlang::list_to_atom_1(Term::NIL), Ok(atom_unchecked("")));
}

#[test]
fn with_improper_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::list::improper(arc_process.clone()),
                |list| {
                    prop_assert_eq!(erlang::list_to_atom_1(list), Err(badarg!().into()));

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
                    let codepoint_terms: Vec<Term> =
                        string.chars().map(|c| arc_process.integer(c)).collect();
                    let list = arc_process.list_from_slice(&codepoint_terms).unwrap();

                    (list, string)
                }),
                |(list, string)| {
                    let atom = atom_unchecked(&string);

                    prop_assert_eq!(erlang::list_to_atom_1(list), Ok(atom));

                    Ok(())
                },
            )
            .unwrap();
    });
}
