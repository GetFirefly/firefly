use super::*;

mod with_big_integer_minuend;
mod with_float_minuend;
mod with_small_integer_minuend;

#[test]
fn with_atom_minuend_errors_badarith() {
    with_minuend_errors_badarith(|_| Term::str_to_atom("minuend", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_minuend_errors_badarith() {
    with_minuend_errors_badarith(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_minuend_errors_badarith() {
    with_minuend_errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_minuend_errors_badarith() {
    with_minuend_errors_badarith(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_local_pid_minuend_errors_badarith() {
    with_minuend_errors_badarith(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_minuend_errors_badarith() {
    with_minuend_errors_badarith(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_minuend_errors_badarith() {
    with_minuend_errors_badarith(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_is_minuend_errors_badarith() {
    with_minuend_errors_badarith(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_minuend_errors_badarith() {
    with_minuend_errors_badarith(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_minuend_errors_badarith() {
    with_minuend_errors_badarith(|process| {
        let original = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(original, 0, 7, 2, 1, &process)
    });
}

fn with_minuend_errors_badarith<M>(minuend: M)
where
    M: FnOnce(&Process) -> Term,
{
    super::errors_badarith(|process| {
        let minuend = minuend(&process);
        let subtrahend = 0.into_process(&process);

        erlang::add_2(minuend, subtrahend, &process)
    });
}
