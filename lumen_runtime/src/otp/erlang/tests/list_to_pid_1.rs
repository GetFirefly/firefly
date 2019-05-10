use super::*;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_errors_badarg() {
    errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_encoding_local_pid() {
    with_process(|process| {
        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<", &process),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<0", &process),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<0.", &process),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<0.1", &process),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<0.1.", &process),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<0.1.2", &process),
            &process
        ));

        assert_eq!(
            erlang::list_to_pid_1(Term::str_to_char_list("<0.1.2>", &process), &process),
            Term::local_pid(1, 2)
        );

        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<0.1.2>?", &process),
            &process
        ));
    })
}

#[test]
fn with_list_encoding_external_pid() {
    with_process(|process| {
        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<", &process),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<1", &process),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<1.", &process),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<1.2", &process),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<1.2.", &process),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<1.2.3", &process),
            &process
        ));

        assert_eq!(
            erlang::list_to_pid_1(Term::str_to_char_list("<1.2.3>", &process), &process),
            Term::external_pid(1, 2, 3, &process)
        );

        assert_badarg!(erlang::list_to_pid_1(
            Term::str_to_char_list("<1.2.3>?", &process),
            &process
        ));
    });
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|process| (integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_is_false() {
    errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|process| bitstring!(1 :: 1, &process));
}

fn errors_badarg<S>(string: S)
where
    S: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::list_to_pid_1(string(&process), &process))
}
