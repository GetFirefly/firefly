use super::*;

mod with_false_left;
mod with_true_left;

#[test]
fn with_atom_left_errors_badarg() {
    with_left_errors_badarg(|_| Term::str_to_atom("left", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_left_errors_badarg() {
    with_left_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_left_errors_badarg() {
    with_left_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_left_errors_badarg() {
    with_left_errors_badarg(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_float_errors_badarg() {
    with_left_errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_left_errors_badarg() {
    with_left_errors_badarg(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_left_errors_badarg() {
    with_left_errors_badarg(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_left_errors_badarg() {
    with_left_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_is_left_errors_badarg() {
    with_left_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_left_errors_badarg() {
    with_left_errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_left_errors_badarg() {
    with_left_errors_badarg(|process| bitstring!(1 ::1, &process));
}

fn with_left_errors_badarg<L>(left: L)
where
    L: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| {
        let left = left(&process);
        let right = 0.into_process(&process);

        erlang::xor_2(left, right)
    });
}
