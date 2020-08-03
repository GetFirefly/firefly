use proptest::prop_assert_eq;

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::binary_to_atom_2::result;
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
fn with_utf8_binary_with_encoding_returns_atom_with_binary_name() {
    run!(
        |arc_process| {
            (
                strategy::term::binary::is_utf8(arc_process.clone()),
                strategy::term::is_encoding(),
            )
        },
        |(binary, encoding)| {
            let byte_vec: Vec<u8> = match binary.decode().unwrap() {
                TypedTerm::HeapBinary(heap_binary) => heap_binary.as_bytes().to_vec(),
                TypedTerm::SubBinary(subbinary) => subbinary.full_byte_iter().collect(),
                TypedTerm::ProcBin(process_binary) => process_binary.as_bytes().to_vec(),
                TypedTerm::BinaryLiteral(process_binary) => process_binary.as_bytes().to_vec(),
                typed_term => panic!("typed_term = {:?}", typed_term),
            };

            let s = std::str::from_utf8(&byte_vec).unwrap();

            prop_assert_eq!(result(binary, encoding), Ok(Atom::str_to_term(s)));

            Ok(())
        },
    );
}
