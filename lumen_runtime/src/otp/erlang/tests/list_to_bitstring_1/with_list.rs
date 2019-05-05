use super::*;

mod with_0_bit_subbinary;
mod with_1_bit_subbinary;
mod with_2_bit_subbinary;
mod with_3_bit_subbinary;
mod with_4_bit_subbinary;
mod with_5_bit_subbinary;
mod with_6_bit_subbinary;
mod with_7_bit_subbinary;
mod with_byte;
mod with_heap_binary;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|process| {
        Term::cons(
            Term::str_to_atom("", DoNotCare).unwrap(),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|process| {
        Term::cons(
            Term::next_local_reference(process),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_empty_list_returns_empty_binary() {
    with_process(|process| {
        let iolist = Term::cons(Term::EMPTY_LIST, Term::EMPTY_LIST, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(Term::slice_to_binary(&[], &process))
        );
    })
}

#[test]
fn with_small_integer_with_byte_overflow_errors_badarg() {
    errors_badarg(|process| Term::cons(256.into_process(&process), Term::EMPTY_LIST, &process))
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|process| {
        Term::cons(
            (crate::integer::small::MAX + 1).into_process(&process),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|process| Term::cons(1.0.into_process(&process), Term::EMPTY_LIST, &process))
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|process| Term::cons(Term::local_pid(0, 0).unwrap(), Term::EMPTY_LIST, &process))
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|process| {
        Term::cons(
            Term::external_pid(1, 0, 0, &process).unwrap(),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|process| {
        Term::cons(
            Term::slice_to_tuple(&[], &process),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|process| {
        Term::cons(
            Term::slice_to_map(&[], &process),
            Term::EMPTY_LIST,
            &process,
        )
    })
}
