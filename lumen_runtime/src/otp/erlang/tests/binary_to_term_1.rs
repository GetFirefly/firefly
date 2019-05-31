use super::*;

use crate::process::IntoProcess;

#[test]
fn without_binary_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_binary(arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        erlang::binary_to_term_1(binary, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_binary_encoding_atom_returns_atom() {
    with_binary_returns_term(
        // :erlang.term_to_binary(:atom)
        vec![131, 100, 0, 4, 97, 116, 111, 109],
        |_| Term::str_to_atom("atom", DoNotCare).unwrap(),
    );
}

#[test]
fn with_binary_encoding_empty_list_returns_empty_list() {
    with_binary_returns_term(
        // :erlang.term_to_binary([])
        vec![131, 106],
        |_| Term::EMPTY_LIST,
    );
}

#[test]
fn with_binary_encoding_list_returns_list() {
    with_binary_returns_term(
        // :erlang.term_to_binary([:zero, 1])
        vec![
            131, 108, 0, 0, 0, 2, 100, 0, 4, 122, 101, 114, 111, 97, 1, 106,
        ],
        |process| {
            Term::cons(
                Term::str_to_atom("zero", DoNotCare).unwrap(),
                Term::cons(1.into_process(process), Term::EMPTY_LIST, process),
                process,
            )
        },
    );
}

#[test]
fn with_binary_encoding_small_integer_returns_small_integer() {
    with_binary_returns_term(
        // :erlang.term_to_binary(0)
        vec![131, 97, 0],
        |process| 0.into_process(process),
    );
}

#[test]
fn with_binary_encoding_integer_returns_integer() {
    with_binary_returns_term(
        // :erlang.term_to_binary(-2147483648)
        vec![131, 98, 128, 0, 0, 0],
        |process| (-2147483648_isize).into_process(process),
    );
}

#[test]
fn with_binary_encoding_new_float_returns_float() {
    with_binary_returns_term(
        // :erlang.term_to_binary(1.0)
        vec![131, 70, 63, 240, 0, 0, 0, 0, 0, 0],
        |process| 1.0.into_process(process),
    );
}

#[test]
fn with_binary_encoding_small_tuple_returns_tuple() {
    with_binary_returns_term(
        // :erlang.term_to_binary({:zero, 1})
        vec![131, 104, 2, 100, 0, 4, 122, 101, 114, 111, 97, 1],
        |process| {
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("zero", DoNotCare).unwrap(),
                    1.into_process(&process),
                ],
                &process,
            )
        },
    );
}

#[test]
fn with_binary_encoding_byte_list_returns_list() {
    with_binary_returns_term(
        // :erlang.term_to_binary([?0, ?1])
        vec![131, 107, 0, 2, 48, 49],
        |process| {
            Term::cons(
                48.into_process(&process),
                Term::cons(49.into_process(&process), Term::EMPTY_LIST, &process),
                &process,
            )
        },
    );
}

#[test]
fn with_binary_encoding_binary_returns_binary() {
    with_binary_returns_term(
        // :erlang.term_to_binary(<<0, 1>>)
        vec![131, 109, 0, 0, 0, 2, 0, 1],
        |process| Term::slice_to_binary(&[0, 1], &process),
    );
}

#[test]
fn with_binary_encoding_small_big_integer_returns_big_integer() {
    with_binary_returns_term(
        // :erlang.term_to_binary(4294967295)
        vec![131, 110, 4, 0, 255, 255, 255, 255],
        |process| 4294967295_usize.into_process(&process),
    );
}

#[test]
fn with_binary_encoding_bit_string_returns_subbinary() {
    with_binary_returns_term(
        // :erlang.term_to_binary(<<1, 2::3>>)
        vec![131, 77, 0, 0, 0, 2, 3, 1, 64],
        |process| {
            Term::subbinary(
                Term::slice_to_binary(&[1, 0b010_00000], &process),
                0,
                0,
                1,
                3,
                &process,
            )
        },
    );
}

#[test]
fn with_binary_encoding_small_atom_utf8_returns_atom() {
    with_binary_returns_term(
        // :erlang.term_to_binary(:"😈")
        vec![131, 119, 4, 240, 159, 152, 136],
        |_| Term::str_to_atom("😈", DoNotCare).unwrap(),
    );
}

fn with_binary_returns_term<T>(byte_vec: Vec<u8>, term: T)
where
    T: Fn(&Process) -> Term,
{
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        erlang::binary_to_term_1(binary, &arc_process),
                        Ok(term(&arc_process))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
