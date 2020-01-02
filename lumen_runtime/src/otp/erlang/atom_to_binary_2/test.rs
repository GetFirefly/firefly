use std::convert::TryInto;

use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::Atom;

use crate::otp::erlang::atom_to_binary_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

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
                    prop_assert_is_not_atom!(native(&arc_process, atom, encoding), atom);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_atom_without_atom_encoding_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term::is_not_atom(arc_process.clone()),
                ),
                |(atom, encoding)| {
                    prop_assert_badarg!(
                        native(&arc_process, atom, encoding),
                        format!("invalid encoding name value: `{}` is not an atom", encoding)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_atom_with_atom_without_name_encoding_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term::atom::is_not_encoding(),
                ),
                |(atom, encoding)| {
                    let encoding_atom: Atom = encoding.try_into().unwrap();

                    prop_assert_badarg!(native(&arc_process, atom, encoding), format!("invalid atom encoding name: '{}' is not one of the supported values (latin1, unicode, or utf8)", encoding_atom.name()));

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
                &(any::<String>(), strategy::term::is_encoding())
                    .prop_map(|(string, encoding)| (Atom::str_to_term(&string), encoding, string)),
                |(atom, encoding, string)| {
                    prop_assert_eq!(
                        native(&arc_process, atom, encoding),
                        Ok(arc_process.binary_from_bytes(string.as_bytes()).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
