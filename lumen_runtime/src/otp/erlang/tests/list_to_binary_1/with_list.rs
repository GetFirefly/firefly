use super::*;

mod with_binary_subbinary;
mod with_byte;
mod with_heap_binary;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|mut process| {
        Term::cons(
            Term::str_to_atom("", DoNotCare).unwrap(),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|mut process| {
        Term::cons(
            Term::local_reference(&mut process),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_empty_list_returns_empty_binary() {
    with_process(|mut process| {
        let iolist = Term::cons(Term::EMPTY_LIST, Term::EMPTY_LIST, &mut process);

        assert_eq!(
            erlang::list_to_binary_1(iolist, &mut process),
            Ok(Term::slice_to_binary(&[], &mut process))
        );
    })
}

#[test]
fn with_small_integer_with_byte_overflow_errors_badarg() {
    errors_badarg(|mut process| {
        Term::cons(
            256.into_process(&mut process),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|mut process| {
        Term::cons(
            (crate::integer::small::MAX + 1).into_process(&mut process),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|mut process| {
        Term::cons(
            1.0.into_process(&mut process),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|mut process| {
        Term::cons(
            Term::local_pid(0, 0).unwrap(),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|mut process| {
        Term::cons(
            Term::external_pid(1, 0, 0, &mut process).unwrap(),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|mut process| {
        Term::cons(
            Term::slice_to_tuple(&[], &mut process),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|mut process| {
        Term::cons(
            Term::slice_to_map(&[], &mut process),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_subbinary_without_bit_count_returns_binary_containing_subbinary_bytes() {
    with_process(|mut process| {
        let original = Term::slice_to_binary(&[0b0111_1111, 0b1000_0000], &mut process);
        let iolist = Term::cons(
            Term::subbinary(original, 0, 1, 1, 0, &mut process),
            Term::EMPTY_LIST,
            &mut process,
        );

        assert_eq!(
            erlang::list_to_binary_1(iolist, &mut process),
            Ok(Term::slice_to_binary(&[0b1111_1111], &mut process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_errors_badarg() {
    errors_badarg(|mut process| {
        let original = Term::slice_to_binary(&[0b0111_1111, 0b1100_0000], &mut process);
        Term::cons(
            Term::subbinary(original, 0, 1, 1, 1, &mut process),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}
