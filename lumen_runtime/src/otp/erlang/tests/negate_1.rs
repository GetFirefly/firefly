use super::*;

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
fn with_small_integer_returns_small_integer() {
    with_process(|process| {
        assert_eq!(
            erlang::negate_1(1.into_process(&process), &process),
            Ok((-1_isize).into_process(&process))
        );
    });
}

#[test]
fn with_big_integer_returns_big_integer() {
    with_process(|process| {
        assert_eq!(
            erlang::negate_1(
                (crate::integer::small::MIN - 1).into_process(&process),
                &process
            ),
            Ok((crate::integer::small::MAX + 2).into_process(&process))
        );
    });
}

#[test]
fn with_float_returns_argument() {
    with_process(|process| {
        assert_eq!(
            erlang::negate_1((-1.0).into_process(&process), &process),
            Ok(1.0.into_process(&process))
        );
    });
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

fn errors_badarith<N>(number: N)
where
    N: FnOnce(&Process) -> Term,
{
    super::errors_badarith(|process| erlang::negate_1(number(&process), &process));
}
