use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_errors_badarg() {
    errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarg() {
    errors_badarg(|process| list_term(&process));
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|process| 0usize.into_process(&process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&process)
    });
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_encoding_atom_returns_atom() {
    with_process(|process| {
        // :erlang.term_to_binary(:atom)
        let heap_binary_term =
            Term::slice_to_binary(&[131, 100, 0, 4, 97, 116, 111, 109], &process);

        assert_eq!(
            erlang::binary_to_term_1(heap_binary_term, &process),
            Ok(Term::str_to_atom("atom", DoNotCare).unwrap())
        );
    });
}

#[test]
fn with_heap_binary_encoding_empty_list_returns_empty_list() {
    with_process(|process| {
        // :erlang.term_to_binary([])
        let heap_binary_term = Term::slice_to_binary(&[131, 106], &process);

        assert_eq!(
            erlang::binary_to_term_1(heap_binary_term, &process),
            Ok(Term::EMPTY_LIST)
        );
    });
}

#[test]
fn with_heap_binary_encoding_list_returns_list() {
    with_process(|process| {
        // :erlang.term_to_binary([:zero, 1])
        let binary = Term::slice_to_binary(
            &[
                131, 108, 0, 0, 0, 2, 100, 0, 4, 122, 101, 114, 111, 97, 1, 106,
            ],
            &process,
        );

        assert_eq!(
            erlang::binary_to_term_1(binary, &process),
            Ok(Term::cons(
                Term::str_to_atom("zero", DoNotCare).unwrap(),
                Term::cons(1.into_process(&process), Term::EMPTY_LIST, &process),
                &process
            ))
        );
    });
}

#[test]
fn with_heap_binary_encoding_small_integer_returns_small_integer() {
    with_process(|process| {
        // :erlang.term_to_binary(0)
        let binary = Term::slice_to_binary(&[131, 97, 0], &process);

        assert_eq!(
            erlang::binary_to_term_1(binary, &process),
            Ok(0.into_process(&process))
        );
    });
}

#[test]
fn with_heap_binary_encoding_integer_returns_integer() {
    with_process(|process| {
        // :erlang.term_to_binary(-2147483648)
        let binary = Term::slice_to_binary(&[131, 98, 128, 0, 0, 0], &process);

        assert_eq!(
            erlang::binary_to_term_1(binary, &process),
            Ok((-2147483648_isize).into_process(&process))
        );
    });
}

#[test]
fn with_heap_binary_encoding_new_float_returns_float() {
    with_process(|process| {
        // :erlang.term_to_binary(1.0)
        let binary = Term::slice_to_binary(&[131, 70, 63, 240, 0, 0, 0, 0, 0, 0], &process);

        assert_eq!(
            erlang::binary_to_term_1(binary, &process),
            Ok(1.0.into_process(&process))
        );
    });
}

#[test]
fn with_heap_binary_encoding_small_tuple_returns_tuple() {
    with_process(|process| {
        // :erlang.term_to_binary({:zero, 1})
        let binary = Term::slice_to_binary(
            &[131, 104, 2, 100, 0, 4, 122, 101, 114, 111, 97, 1],
            &process,
        );

        assert_eq!(
            erlang::binary_to_term_1(binary, &process),
            Ok(Term::slice_to_tuple(
                &[
                    Term::str_to_atom("zero", DoNotCare).unwrap(),
                    1.into_process(&process)
                ],
                &process
            ))
        );
    });
}

#[test]
fn with_heap_binary_encoding_byte_list_returns_list() {
    with_process(|process| {
        // :erlang.term_to_binary([?0, ?1])
        let binary = Term::slice_to_binary(&[131, 107, 0, 2, 48, 49], &process);

        assert_eq!(
            erlang::binary_to_term_1(binary, &process),
            Ok(Term::cons(
                48.into_process(&process),
                Term::cons(49.into_process(&process), Term::EMPTY_LIST, &process),
                &process
            ))
        );
    });
}

#[test]
fn with_heap_binary_encoding_binary_returns_binary() {
    with_process(|process| {
        // :erlang.term_to_binary(<<0, 1>>)
        let heap_binary_term = Term::slice_to_binary(&[131, 109, 0, 0, 0, 2, 0, 1], &process);

        assert_eq!(
            erlang::binary_to_term_1(heap_binary_term, &process),
            Ok(Term::slice_to_binary(&[0, 1], &process))
        );
    });
}

#[test]
fn with_heap_binary_encoding_small_big_integer_returns_big_integer() {
    with_process(|process| {
        // :erlang.term_to_binary(4294967295)
        let heap_binary_term =
            Term::slice_to_binary(&[131, 110, 4, 0, 255, 255, 255, 255], &process);

        assert_eq!(
            erlang::binary_to_term_1(heap_binary_term, &process),
            Ok(4294967295_usize.into_process(&process))
        );
    });
}

#[test]
fn with_heap_binary_encoding_bit_string_returns_subbinary() {
    with_process(|process| {
        // :erlang.term_to_binary(<<1, 2::3>>)
        let heap_binary_term = Term::slice_to_binary(&[131, 77, 0, 0, 0, 2, 3, 1, 64], &process);

        assert_eq!(
            erlang::binary_to_term_1(heap_binary_term, &process),
            Ok(Term::subbinary(
                Term::slice_to_binary(&[1, 0b010_00000], &process),
                0,
                0,
                1,
                3,
                &process
            ))
        );
    });
}

#[test]
fn with_heap_binary_encoding_small_atom_utf8_returns_atom() {
    with_process(|process| {
        // :erlang.term_to_binary(:"😈")
        let heap_binary_term = Term::slice_to_binary(&[131, 119, 4, 240, 159, 152, 136], &process);

        assert_eq!(
            erlang::binary_to_term_1(heap_binary_term, &process),
            Ok(Term::str_to_atom("😈", DoNotCare).unwrap())
        );
    });
}

#[test]
fn with_subbinary_encoding_atom_returns_atom() {
    with_process(|process| {
        // <<1::1, :erlang.term_to_binary(:atom) :: binary>>
        let original_term =
            Term::slice_to_binary(&[193, 178, 0, 2, 48, 186, 55, 182, 0b1000_0000], &process);
        let subbinary_term = Term::subbinary(original_term, 0, 1, 8, 0, &process);

        assert_eq!(
            erlang::binary_to_term_1(subbinary_term, &process),
            Ok(Term::str_to_atom("atom", DoNotCare).unwrap())
        );
    });
}

#[test]
fn with_subbinary_encoding_empty_list_returns_empty_list() {
    with_process(|process| {
        // <<1::1, :erlang.term_to_binary([]) :: binary>>
        let original_term = Term::slice_to_binary(&[193, 181, 0b0000_0000], &process);
        let subbinary_term = Term::subbinary(original_term, 0, 1, 2, 0, &process);

        assert_eq!(
            erlang::binary_to_term_1(subbinary_term, &process),
            Ok(Term::EMPTY_LIST)
        );
    });
}

#[test]
fn with_subbinary_encoding_list_returns_list() {
    with_process(|process| {
        // <<1::1, :erlang.term_to_binary([:zero, 1]) :: binary>>
        let original_term = Term::slice_to_binary(
            &[
                193, 182, 0, 0, 0, 1, 50, 0, 2, 61, 50, 185, 55, 176, 128, 181, 0,
            ],
            &process,
        );
        let subbinary_term = Term::subbinary(original_term, 0, 1, 16, 0, &process);

        assert_eq!(
            erlang::binary_to_term_1(subbinary_term, &process),
            Ok(Term::cons(
                Term::str_to_atom("zero", DoNotCare).unwrap(),
                Term::cons(1.into_process(&process), Term::EMPTY_LIST, &process),
                &process
            ))
        );
    });
}

#[test]
fn with_subbinary_encoding_small_integer_returns_small_integer() {
    with_process(|process| {
        // <<1::1, :erlang.term_to_binary(0) :: binary>>
        let original_term = Term::slice_to_binary(&[193, 176, 128, 0], &process);
        let subbinary_term = Term::subbinary(original_term, 0, 1, 3, 0, &process);

        assert_eq!(
            erlang::binary_to_term_1(subbinary_term, &process),
            Ok(0.into_process(&process))
        );
    });
}

#[test]
fn with_subbinary_encoding_integer_returns_integer() {
    with_process(|process| {
        // <<1::1, :erlang.term_to_binary(-2147483648) :: binary>>
        let original_term = Term::slice_to_binary(&[193, 177, 64, 0, 0, 0, 0], &process);
        let subbinary_term = Term::subbinary(original_term, 0, 1, 6, 0, &process);

        assert_eq!(
            erlang::binary_to_term_1(subbinary_term, &process),
            Ok((-2147483648_isize).into_process(&process))
        );
    });
}

#[test]
fn with_subbinary_encoding_new_float_returns_float() {
    with_process(|process| {
        // <<1::1, :erlang.term_to_binary(1.0) :: binary>>
        let original_term =
            Term::slice_to_binary(&[193, 163, 31, 248, 0, 0, 0, 0, 0, 0, 0], &process);
        let subbinary_term = Term::subbinary(original_term, 0, 1, 10, 0, &process);

        assert_eq!(
            erlang::binary_to_term_1(subbinary_term, &process),
            Ok(1.0.into_process(&process))
        );
    });
}

#[test]
fn with_subbinary_encoding_small_tuple_returns_tuple() {
    with_process(|process| {
        // <<1::1, :erlang.term_to_binary({:zero, 1}) :: binary>>
        let original_term = Term::slice_to_binary(
            &[
                193,
                180,
                1,
                50,
                0,
                2,
                61,
                50,
                185,
                55,
                176,
                128,
                0b1000_0000,
            ],
            &process,
        );
        let subbinary_term = Term::subbinary(original_term, 0, 1, 12, 0, &process);

        assert_eq!(
            erlang::binary_to_term_1(subbinary_term, &process),
            Ok(Term::slice_to_tuple(
                &[
                    Term::str_to_atom("zero", DoNotCare).unwrap(),
                    1.into_process(&process)
                ],
                &process
            ))
        );
    });
}

#[test]
fn with_subbinary_encoding_byte_list_returns_list() {
    with_process(|process| {
        // <<1::1, :erlang.term_to_binary([?0, ?1]) :: binary>>
        let original_term =
            Term::slice_to_binary(&[193, 181, 128, 1, 24, 24, 0b1000_0000], &process);
        let subbinary_term = Term::subbinary(original_term, 0, 1, 6, 0, &process);

        assert_eq!(
            erlang::binary_to_term_1(subbinary_term, &process),
            Ok(Term::cons(
                48.into_process(&process),
                Term::cons(49.into_process(&process), Term::EMPTY_LIST, &process),
                &process
            ))
        );
    });
}

#[test]
fn with_subbinary_encoding_binary_returns_binary() {
    with_process(|process| {
        // <<1::1, :erlang.term_to_binary(<<0, 1>>) :: binary>>
        let original_term =
            Term::slice_to_binary(&[193, 182, 128, 0, 0, 1, 0, 0, 0b1000_0000], &process);
        let subbinary_term = Term::subbinary(original_term, 0, 1, 8, 0, &process);

        assert_eq!(
            erlang::binary_to_term_1(subbinary_term, &process),
            Ok(Term::slice_to_binary(&[0, 1], &process))
        );
    });
}

#[test]
fn with_subbinary_encoding_small_big_integer_returns_big_integer() {
    with_process(|process| {
        // <<1::1, :erlang.term_to_binary(4294967295) :: binary>>
        let original_term =
            Term::slice_to_binary(&[193, 183, 2, 0, 127, 255, 255, 255, 0b1000_0000], &process);
        let subbinary_term = Term::subbinary(original_term, 0, 1, 8, 0, &process);

        assert_eq!(
            erlang::binary_to_term_1(subbinary_term, &process),
            Ok(4294967295_usize.into_process(&process))
        );
    });
}

#[test]
fn with_subbinary_encoding_bit_string_returns_subbinary() {
    with_process(|process| {
        // <<1::1, :erlang.term_to_binary(<<1, 2::3>>) :: binary>>
        let original_term =
            Term::slice_to_binary(&[193, 166, 128, 0, 0, 1, 1, 128, 160, 0], &process);
        let subbinary_term = Term::subbinary(original_term, 0, 1, 9, 0, &process);

        assert_eq!(
            erlang::binary_to_term_1(subbinary_term, &process),
            Ok(Term::subbinary(
                Term::slice_to_binary(&[1, 0b010_00000], &process),
                0,
                0,
                1,
                3,
                &process
            ))
        );
    });
}

#[test]
fn with_subbinary_encoding_small_atom_utf8_returns_atom() {
    with_process(|process| {
        // <<1::1, :erlang.term_to_binary(:"😈") :: binary>>
        let original_term = Term::slice_to_binary(&[193, 187, 130, 120, 79, 204, 68, 0], &process);
        let subbinary_term = Term::subbinary(original_term, 0, 1, 7, 0, &process);

        assert_eq!(
            erlang::binary_to_term_1(subbinary_term, &process),
            Ok(Term::str_to_atom("😈", DoNotCare).unwrap())
        );
    });
}

fn errors_badarg<F>(binary: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::binary_to_term_1(binary(&process), &process));
}
