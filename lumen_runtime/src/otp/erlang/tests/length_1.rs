use super::*;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_is_zero() {
    with_process(|process| {
        let zero_term = 0.into_process(&process);

        assert_eq!(erlang::length_1(Term::EMPTY_LIST, &process), Ok(zero_term));
    });
}

#[test]
fn with_improper_list_errors_badarg() {
    errors_badarg(|process| {
        let head_term = Term::str_to_atom("head", DoNotCare).unwrap();
        let tail_term = Term::str_to_atom("tail", DoNotCare).unwrap();
        Term::cons(head_term, tail_term, &process)
    });
}

#[test]
fn with_list_is_length_1() {
    with_process(|process| {
        let list_term = (0..=2).rfold(Term::EMPTY_LIST, |acc, i| {
            Term::cons(i.into_process(&process), acc, &process)
        });

        assert_eq!(
            erlang::length_1(list_term, &process),
            Ok(3.into_process(&process))
        );
    });
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|process| (integer::small::MAX + 1).into_process(&process));
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
    super::errors_badarg(|process| erlang::length_1(list(&process), &process));
}
