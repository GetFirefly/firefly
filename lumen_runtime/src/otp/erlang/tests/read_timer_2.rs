use super::*;

mod with_empty_list_options;
mod with_list_options;

#[test]
fn with_atom_options_errors_badarg() {
    with_options_errors_badarg(|_| Term::str_to_atom("nosuspend", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_options_errors_badarg() {
    with_options_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_small_integer_options_errors_badarg() {
    with_options_errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_options_errors_badarg() {
    with_options_errors_badarg(|process| (crate::integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_options_errors_badarg() {
    with_options_errors_badarg(|process| 0.0.into_process(&process));
}

#[test]
fn with_local_pid_options_errors_badarg() {
    with_options_errors_badarg(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_options_errors_badarg() {
    with_options_errors_badarg(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_options_errors_badarg() {
    with_options_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_options_errors_badarg() {
    with_options_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_options_errors_badarg() {
    with_options_errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_options_errors_badarg() {
    with_options_errors_badarg(|process| bitstring!(1 :: 1, &process));
}

fn with_options_errors_badarg<O>(options: O)
where
    O: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let timer_reference = Term::next_local_reference(process);
        let options = options(process);

        assert_badarg!(erlang::read_timer_2(timer_reference, options, process));
    });
}
