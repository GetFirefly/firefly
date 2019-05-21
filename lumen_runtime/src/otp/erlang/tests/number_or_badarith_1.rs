use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

#[test]
fn with_atom_errors_badarith() {
    errors_badarith(|_| Term::str_to_atom("number", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarith() {
    errors_badarith(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_errors_badarith() {
    errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarith() {
    errors_badarith(|process| list_term(&process));
}

#[test]
fn with_small_integer_returns_argument() {
    returns_term(|process| 0usize.into_process(&process));
}

#[test]
fn with_big_integer_returns_argument() {
    returns_term(|process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&process)
    });
}

#[test]
fn with_float_returns_argument() {
    returns_term(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_badarith() {
    errors_badarith(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarith() {
    errors_badarith(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_errors_badarith() {
    errors_badarith(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_errors_badarith() {
    errors_badarith(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_errors_badarith() {
    errors_badarith(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_errors_badarith() {
    errors_badarith(|process| {
        let original = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(original, 0, 7, 2, 1, &process)
    });
}

fn errors_badarith<T>(term: T)
where
    T: FnOnce(&Process) -> Term,
{
    super::errors_badarith(|process| erlang::number_or_badarith_1(term(&process)));
}

fn returns_term<T>(term: T)
where
    T: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let term = term(&process);

        assert_eq!(erlang::number_or_badarith_1(term), Ok(term))
    })
}
