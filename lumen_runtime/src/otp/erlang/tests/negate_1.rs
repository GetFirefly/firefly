use super::*;

use crate::process::IntoProcess;

#[test]
fn with_atom_errors_badarith() {
    errors_badarith(|_| Term::str_to_atom("number", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarith() {
    errors_badarith(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_errors_badarith() {
    errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarith() {
    errors_badarith(|mut process| list_term(&mut process));
}

#[test]
fn with_small_integer_returns_small_integer() {
    with_process(|mut process| {
        assert_eq!(
            erlang::negate_1(1.into_process(&mut process), &mut process),
            Ok((-1_isize).into_process(&mut process))
        );
    });
}

#[test]
fn with_big_integer_returns_big_integer() {
    with_process(|mut process| {
        assert_eq!(
            erlang::negate_1(
                (crate::integer::small::MIN - 1).into_process(&mut process),
                &mut process
            ),
            Ok((crate::integer::small::MAX + 2).into_process(&mut process))
        );
    });
}

#[test]
fn with_float_returns_argument() {
    with_process(|mut process| {
        assert_eq!(
            erlang::negate_1((-1.0).into_process(&mut process), &mut process),
            Ok(1.0.into_process(&mut process))
        );
    });
}

#[test]
fn with_local_pid_errors_badarith() {
    errors_badarith(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarith() {
    errors_badarith(|mut process| Term::external_pid(1, 0, 0, &mut process).unwrap());
}

#[test]
fn with_tuple_errors_badarith() {
    errors_badarith(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_errors_badarith() {
    errors_badarith(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_errors_badarith() {
    errors_badarith(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_errors_badarith() {
    errors_badarith(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        Term::subbinary(original, 0, 7, 2, 1, &mut process)
    });
}

fn errors_badarith<N>(number: N)
where
    N: FnOnce(&mut Process) -> Term,
{
    super::errors_badarith(|mut process| erlang::negate_1(number(&mut process), &mut process));
}
