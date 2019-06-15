use super::*;

#[test]
fn with_atom_right_errors_badarg() {
    with_right_errors_badarg(|_| Term::str_to_atom("right", DoNotCare).unwrap());
}

#[test]
fn with_false_right_returns_false() {
    with(|left, _| {
        assert_eq!(erlang::xor_2(left, false.into()), Ok(false.into()));
    });
}

#[test]
fn with_true_right_returns_true() {
    with(|left, _| {
        assert_eq!(erlang::xor_2(left, true.into()), Ok(true.into()));
    });
}

#[test]
fn with_local_reference_right_errors_badarg() {
    with_right_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_right_errors_badarg() {
    with_right_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_right_errors_badarg() {
    with_right_errors_badarg(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_right_errors_badarg() {
    with_right_errors_badarg(|process| 1.into_process(&process))
}

#[test]
fn with_big_integer_right_errors_badarg() {
    with_right_errors_badarg(|process| (crate::integer::small::MAX + 1).into_process(&process))
}

#[test]
fn with_float_right_errors_badarg() {
    with_right_errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_right_errors_badarg() {
    with_right_errors_badarg(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_right_errors_badarg() {
    with_right_errors_badarg(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_right_errors_badarg() {
    with_right_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_is_right_errors_badarg() {
    with_right_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_right_errors_badarg() {
    with_right_errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_right_errors_badarg() {
    with_right_errors_badarg(|process| bitstring!(1 :: 1, &process));
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let left = false.into();

        f(left, &process)
    })
}

fn with_right_errors_badarg<M>(right: M)
where
    M: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| {
        let left = false.into();
        let right = right(&process);

        erlang::xor_2(left, right)
    });
}
