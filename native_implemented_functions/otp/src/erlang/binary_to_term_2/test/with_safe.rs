use super::*;

#[test]
fn with_binary_encoding_atom_that_does_not_exist_errors_badarg() {
    // :erlang.term_to_binary(:non_existent_0)
    tried_to_convert_to_an_atom_that_doesnt_exist(vec![
        131, 100, 0, 14, 110, 111, 110, 95, 101, 120, 105, 115, 116, 101, 110, 116, 95, 48,
    ]);
}

#[test]
fn with_binary_encoding_list_containing_atom_that_does_not_exist_errors_badarg() {
    // :erlang.term_to_binary([:non_existent_1])
    tried_to_convert_to_an_atom_that_doesnt_exist(vec![
        131, 108, 0, 0, 0, 1, 100, 0, 14, 110, 111, 110, 95, 101, 120, 105, 115, 116, 101, 110,
        116, 95, 49, 106,
    ]);
}

#[test]
fn with_binary_encoding_small_tuple_containing_atom_that_does_not_exist_errors_badarg() {
    // :erlang.term_to_binary({:non_existent_2})
    tried_to_convert_to_an_atom_that_doesnt_exist(vec![
        131, 104, 1, 100, 0, 14, 110, 111, 110, 95, 101, 120, 105, 115, 116, 101, 110, 116, 95, 50,
    ]);
}

#[test]
fn with_binary_encoding_small_atom_utf8_that_does_not_exist_errors_badarg() {
    // :erlang.term_to_binary(:"non_existent_3_😈")
    tried_to_convert_to_an_atom_that_doesnt_exist(vec![
        131, 119, 19, 110, 111, 110, 95, 101, 120, 105, 115, 116, 101, 110, 116, 95, 51, 95, 240,
        159, 152, 136,
    ]);
}

fn options(process: &Process) -> Term {
    process.cons(Atom::str_to_term("safe"), Term::NIL).unwrap()
}

fn tried_to_convert_to_an_atom_that_doesnt_exist(byte_vec: Vec<u8>) {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::containing_bytes(byte_vec.clone(), arc_process.clone()),
            )
        },
        |(arc_process, binary)| {
            prop_assert_badarg!(
                result(&arc_process, binary, options(&arc_process)),
                "tried to convert to an atom that doesn't exist"
            );

            Ok(())
        },
    );
}
