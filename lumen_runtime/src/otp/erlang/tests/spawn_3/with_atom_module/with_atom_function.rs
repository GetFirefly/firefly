use super::*;

mod with_empty_list_arguments;
mod with_proper_list_arguments;

#[test]
fn with_atom_argumetns_errors_badarg() {
    with_arguments_errors_badarg(|_| Term::str_to_atom("arguments", DoNotCare).unwrap());
}

#[test]
fn with_small_integer_arguments_errors_badarg() {
    with_arguments_errors_badarg(|process| 0.into_process(process));
}

#[test]
fn with_float_arguments_errors_badarg() {
    with_arguments_errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_big_integer_arguments_errors_badarg() {
    with_arguments_errors_badarg(|process| (integer::small::MAX + 1).into_process(process));
}

#[test]
fn with_local_reference_arguments_errors_badarg() {
    with_arguments_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_local_pid_arguments_errors_badarg() {
    with_arguments_errors_badarg(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_arguments_errors_badarg() {
    with_arguments_errors_badarg(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_arguments_errors_badarg() {
    with_arguments_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_is_arguments_errors_badarg() {
    with_arguments_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_improper_list_arguments_errors_badarg() {
    with_arguments_errors_badarg(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_heap_binary_arguments_errors_badarg() {
    with_arguments_errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_arguments_errors_badarg() {
    with_arguments_errors_badarg(|process| bitstring!(1 ::1, &process));
}

fn with_arguments_errors_badarg<M>(arguments: M)
where
    M: FnOnce(&Process) -> Term,
{
    errors_badarg(|process| {
        let module = Term::str_to_atom("module", DoNotCare).unwrap();
        let function = Term::str_to_atom("function", DoNotCare).unwrap();

        erlang::spawn_3(module, function, arguments(process), process)
    })
}
