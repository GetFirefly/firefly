use super::*;

mod with_big_integer_left;
mod with_small_integer_left;

#[test]
fn with_atom_left_errors_badarith() {
    with_left_errors_badarith(|_| Term::str_to_atom("left", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_left_errors_badarith() {
    with_left_errors_badarith(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_left_errors_badarith() {
    with_left_errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_left_errors_badarith() {
    with_left_errors_badarith(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_float_errors_badarith() {
    with_left_errors_badarith(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_left_errors_badarith() {
    with_left_errors_badarith(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_left_errors_badarith() {
    with_left_errors_badarith(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_left_errors_badarith() {
    with_left_errors_badarith(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_is_left_errors_badarith() {
    with_left_errors_badarith(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_left_errors_badarith() {
    with_left_errors_badarith(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_left_errors_badarith() {
    with_left_errors_badarith(|process| bitstring!(1 ::1, &process));
}

fn with_left_errors_badarith<L>(left: L)
where
    L: FnOnce(&Process) -> Term,
{
    super::errors_badarith(|process| {
        let left = left(&process);
        let right = 0.into_process(&process);

        erlang::bor_2(left, right, &process)
    });
}
