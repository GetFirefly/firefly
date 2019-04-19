use super::*;

#[test]
fn with_atom_divisor_errors_badarith() {
    with_divisor_errors_badarith(|_| Term::str_to_atom("dividend", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_divisor_errors_badarith() {
    with_divisor_errors_badarith(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_divisor_errors_badarith() {
    with_divisor_errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_divisor_errors_badarith() {
    with_divisor_errors_badarith(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_zero_errors_badarith() {
    with_divisor_errors_badarith(|process| 0.into_process(&process));
}

#[test]
fn with_small_integer_divisor_errors_badarith() {
    with_divisor_errors_badarith(|process| crate::integer::small::MIN.into_process(&process));
}

#[test]
fn with_big_integer_divisor_errors_badarith() {
    with_divisor_errors_badarith(|process| (crate::integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_divisor_errors_badarith() {
    with_divisor_errors_badarith(|process| 4.0.into_process(&process));
}

#[test]
fn with_local_pid_divisor_errors_badarith() {
    with_divisor_errors_badarith(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_divisor_errors_badarith() {
    with_divisor_errors_badarith(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_divisor_errors_badarith() {
    with_divisor_errors_badarith(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_is_divisor_errors_badarith() {
    with_divisor_errors_badarith(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_divisor_errors_badarith() {
    with_divisor_errors_badarith(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_divisor_errors_badarith() {
    with_divisor_errors_badarith(|process| {
        let original = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(original, 0, 7, 2, 1, &process)
    });
}

fn with_divisor_errors_badarith<M>(divisor: M)
where
    M: FnOnce(&Process) -> Term,
{
    super::errors_badarith(|process| {
        let dividend: Term = 2.0.into_process(&process);
        let divisor = divisor(&process);

        erlang::rem_2(dividend, divisor, &process)
    });
}
