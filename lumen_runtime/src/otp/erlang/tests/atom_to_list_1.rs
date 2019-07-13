use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_atom(arc_process.clone()), |atom| {
                prop_assert_eq!(
                    erlang::atom_to_list_1(atom, &arc_process),
                    Err(badarg!().into())
                );

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
                &any::<String>().prop_map(|string| (atom_unchecked(&string), string)),
                |(atom, string)| {
                    let mut heap = arc_process.acquire_heap();

                    let codepoint_terms: Vec<Term> =
                        string.chars().map(|c| heap.integer(c)).collect();

                    prop_assert_eq!(
                        erlang::atom_to_list_1(atom, &arc_process),
                        Ok(arc_process.list_from_slice(&codepoint_terms).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
