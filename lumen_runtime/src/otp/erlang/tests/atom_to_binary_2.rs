use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term::is_encoding(),
                ),
                |(atom, encoding)| {
                    prop_assert_eq!(
                        erlang::atom_to_binary_2(atom, encoding, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_atom_without_encoding_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term::is_not_encoding(arc_process.clone()),
                ),
                |(atom, encoding)| {
                    prop_assert_eq!(
                        erlang::atom_to_binary_2(atom, encoding, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_atom_with_encoding_atom_returns_name_in_binary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(any::<String>(), strategy::term::is_encoding()).prop_map(|(string, encoding)| {
                    (
                        Term::str_to_atom(&string, DoNotCare).unwrap(),
                        encoding,
                        string,
                    )
                }),
                |(atom, encoding, string)| {
                    prop_assert_eq!(
                        erlang::atom_to_binary_2(atom, encoding, &arc_process),
                        Ok(Term::slice_to_binary(string.as_bytes(), &arc_process))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
