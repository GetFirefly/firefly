use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

#[test]
fn without_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term::is_not_atom(arc_process.clone()), encoding()),
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
fn with_atom_without_encoding_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term::is_not_atom(arc_process.clone()),
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
fn with_atom_with_invalid_encoding_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term::atom().prop_filter(
                        "Atom must not be a valid encoding",
                        |encoding| match unsafe { encoding.atom_to_string() }.as_ref().as_ref() {
                            "latin1" | "unicode" | "utf8" => false,
                            _ => true,
                        },
                    ),
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
                &(any::<String>(), encoding()).prop_map(|(string, encoding)| {
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

fn encoding() -> impl Strategy<Value = Term> {
    prop_oneof![
        Just(Term::str_to_atom("latin1", DoNotCare).unwrap()),
        Just(Term::str_to_atom("unicode", DoNotCare).unwrap()),
        Just(Term::str_to_atom("utf8", DoNotCare).unwrap())
    ]
}
