use super::*;

use num_traits::Num;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
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
fn with_empty_tuple_is_empty_list() {
    with_process(|process| {
        let tuple_term = Term::slice_to_tuple(&[], &process);

        assert_eq!(
            erlang::tuple_to_list_1(tuple_term, &process),
            Ok(Term::EMPTY_LIST)
        );
    });
}

#[test]
fn with_nonempty_tuple_is_list() {
    with_process(|process| {
        let first_element = 1.into_process(&process);
        let second_element = 2.into_process(&process);
        let third_element = 3.into_process(&process);
        let tuple_term =
            Term::slice_to_tuple(&[first_element, second_element, third_element], &process);

        assert_eq!(
            erlang::tuple_to_list_1(tuple_term, &process),
            Ok(Term::cons(
                first_element,
                Term::cons(
                    second_element,
                    Term::cons(third_element, Term::EMPTY_LIST, &process),
                    &process,
                ),
                &process,
            ))
        );
    });
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|process| {
        let original = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(original, 0, 7, 2, 1, &process)
    });
}

fn errors_badarg<F>(bit_string: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::tuple_to_list_1(bit_string(&process), &process));
}
