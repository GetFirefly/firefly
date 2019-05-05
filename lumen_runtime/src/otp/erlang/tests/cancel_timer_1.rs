use super::*;

mod with_local_reference;

#[test]
fn with_small_integer_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| 1.into_process(process))
}

#[test]
fn with_float_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| 1.0.into_process(process));
}

#[test]
fn with_big_integer_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| (integer::small::MAX + 1).into_process(process));
}

#[test]
fn with_atom_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_pid_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| Term::external_pid(1, 0, 0, process).unwrap());
}

#[test]
fn with_tuple_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| Term::slice_to_tuple(&[], process));
}

#[test]
fn with_map_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| Term::slice_to_map(&[], process));
}

#[test]
fn with_empty_list_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| list_term(process));
}

#[test]
fn with_heap_binary_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| Term::slice_to_binary(&[1], process));
}

#[test]
fn with_subbinary_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| {
        let original = Term::slice_to_binary(&[0, 1], process);
        Term::subbinary(original, 1, 0, 1, 0, process)
    });
}

fn with_timer_reference_errors_badarg<T>(timer_reference: T)
where
    T: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        assert_badarg!(erlang::cancel_timer_1(timer_reference(process), process));
    });
}
