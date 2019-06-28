use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_atom(arc_process.clone()), |atom| {
                prop_assert_eq!(erlang::atom_to_list_1(atom, &arc_process), Err(badarg!()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_atom_returns_chars_in_list() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &any::<String>()
                    .prop_map(|string| (Term::str_to_atom(&string, DoNotCare).unwrap(), string)),
                |(atom, string)| {
                    let codepoint_terms: Vec<Term> = string
                        .chars()
                        .map(|c| c.into_process(&arc_process))
                        .collect();

                    prop_assert_eq!(
                        erlang::atom_to_list_1(atom, &arc_process),
                        Ok(Term::slice_to_list(&codepoint_terms, &arc_process))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
