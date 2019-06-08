use proptest::strategy::Strategy;

use super::*;

#[test]
fn without_tuple_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_tuple(arc_process.clone()),
                    strategy::term::atom(),
                ),
                |(tuple, record_tag)| {
                    prop_assert_eq!(erlang::is_record_2(tuple, record_tag), Ok(false.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_tuple_without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term::is_not_atom(arc_process.clone()),
                ),
                |(tuple, record_tag)| {
                    prop_assert_eq!(erlang::is_record_2(tuple, record_tag), Err(badarg!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_tuple_with_atom_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::atom(),
                |record_tag| {
                    let tuple = Term::slice_to_tuple(&[], &arc_process);

                    prop_assert_eq!(erlang::is_record_2(tuple, record_tag), Ok(false.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_non_empty_tuple_with_atom_without_record_tag_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_atom(arc_process.clone()),
                    proptest::collection::vec(
                        strategy::term(arc_process.clone()),
                        strategy::size_range(),
                    ),
                    strategy::term::atom(),
                )
                    .prop_map(|(first_element, mut tail_element_vec, atom)| {
                        tail_element_vec.insert(0, first_element);

                        (Term::slice_to_tuple(&tail_element_vec, &arc_process), atom)
                    }),
                |(tuple, record_tag)| {
                    prop_assert_eq!(erlang::is_record_2(tuple, record_tag), Ok(false.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}
