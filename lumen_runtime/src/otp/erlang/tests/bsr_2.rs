use super::*;

mod with_big_integer_integer;
mod with_small_integer_integer;

#[test]
fn with_atom_integer_errors_badarith() {
    with_integer_errors_badarith(|_| Term::str_to_atom("integer", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_integer_errors_badarith() {
    with_integer_errors_badarith(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_integer_errors_badarith() {
    with_integer_errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_integer_errors_badarith() {
    with_integer_errors_badarith(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_float_errors_badarith() {
    with_integer_errors_badarith(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_integer_errors_badarith() {
    with_integer_errors_badarith(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_integer_errors_badarith() {
    with_integer_errors_badarith(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_integer_errors_badarith() {
    with_integer_errors_badarith(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_is_integer_errors_badarith() {
    with_integer_errors_badarith(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_integer_errors_badarith() {
    with_integer_errors_badarith(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_integer_errors_badarith() {
    with_integer_errors_badarith(|process| bitstring!(1 ::1, &process));
}

fn with_integer_errors_badarith<I>(integer: I)
where
    I: FnOnce(&Process) -> Term,
{
    super::errors_badarith(|process| {
        let integer = integer(&process);
        let shift = 0.into_process(&process);

        erlang::bsr_2(integer, shift, &process)
    });
}
