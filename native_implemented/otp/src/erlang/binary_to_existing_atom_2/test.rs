use proptest::prop_assert_eq;

use firefly_rt::term::Term;

use crate::erlang::binary_to_existing_atom_2::result;
use crate::test::strategy;

#[test]
fn without_binary_errors_badarg() {
    crate::test::without_binary_with_encoding_is_not_binary(file!(), result);
}

#[test]
fn with_binary_without_atom_encoding_errors_badarg() {
    crate::test::with_binary_without_atom_encoding_errors_badarg(file!(), result);
}

#[test]
fn with_binary_with_atom_without_name_encoding_errors_badarg() {
    crate::test::with_binary_with_atom_without_name_encoding_errors_badarg(file!(), result);
}

#[test]
fn with_utf8_binary_with_valid_encoding_without_existing_atom_errors_badarg() {
    run!(
        |arc_process| {
            (
                strategy::term::binary::containing_bytes(
                    strategy::term::non_existent_atom("binary_to_existing_atom")
                        .as_bytes()
                        .to_owned(),
                    arc_process.clone(),
                ),
                strategy::term::is_encoding(),
            )
        },
        |(binary, encoding)| {
            prop_assert_badarg!(
                result(binary, encoding),
                "tried to convert to an atom that doesn't exist"
            );

            Ok(())
        },
    );
}

#[test]
fn with_utf8_binary_with_valid_encoding_with_existing_atom_returns_atom() {
    run!(
        |arc_process| {
            (
                strategy::term::binary::is_utf8(arc_process.clone()),
                strategy::term::is_encoding(),
            )
        },
        |(binary, encoding)| {
            let byte_vec: Vec<u8> = match binary {
                Term::HeapBinary(heap_binary) => heap_binary.as_bytes().to_vec(),
                Term::RcBinary(process_binary) => process_binary.as_bytes().to_vec(),
                Term::ConstantBinary(process_binary) => process_binary.as_bytes().to_vec(),
                Term::RefBinary(subbinary) => subbinary.full_byte_iter().collect(),
                typed_term => panic!("typed_term = {:?}", typed_term),
            };

            let s = std::str::from_utf8(&byte_vec).unwrap();
            let existing_atom = Atom::str_to_term(s);

            prop_assert_eq!(result(binary, encoding), Ok(existing_atom));

            Ok(())
        },
    );
}
