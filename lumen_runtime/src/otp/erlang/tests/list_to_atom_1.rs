use super::*;

use num_traits::Num;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("list", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_returns_empty_atom() {
    assert_eq!(
        erlang::list_to_atom_1(Term::EMPTY_LIST),
        Ok(Term::str_to_atom("", DoNotCare).unwrap())
    );
}

#[test]
fn with_improper_list_errors_badarg() {
    errors_badarg(|process| {
        Term::cons(
            'a'.into_process(&process),
            'b'.into_process(&process),
            &process,
        )
    });
}

#[test]
fn with_list_encoding_utf8() {
    with_process(|process| {
        assert_eq!(
            erlang::list_to_atom_1(Term::str_to_char_list("atom", &process)),
            Ok(Term::str_to_atom("atom", DoNotCare).unwrap())
        );
        assert_eq!(
            erlang::list_to_atom_1(Term::str_to_char_list("José", &process)),
            Ok(Term::str_to_atom("José", DoNotCare).unwrap())
        );
        assert_eq!(
            erlang::list_to_atom_1(Term::str_to_char_list("😈", &process)),
            Ok(Term::str_to_atom("😈", DoNotCare).unwrap())
        );
    });
}

#[test]
fn with_list_not_encoding_ut8() {
    errors_badarg(|process| {
        Term::cons(
            // from https://doc.rust-lang.org/std/char/fn.from_u32.html
            0x110000.into_process(&process),
            Term::EMPTY_LIST,
            &process,
        )
    });
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|process| 0.into_process(&process));
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
fn with_heap_binary_is_false() {
    errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_is_false() {
    errors_badarg(|process| {
        let original = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(original, 0, 7, 2, 1, &process)
    });
}

fn errors_badarg<F>(list: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::list_to_atom_1(list(&process)));
}
