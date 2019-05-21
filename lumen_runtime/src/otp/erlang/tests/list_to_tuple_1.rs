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
fn with_empty_list_returns_empty_tuple() {
    with_process(|process| {
        let list = Term::EMPTY_LIST;

        assert_eq!(
            erlang::list_to_tuple_1(list, &process),
            Ok(Term::slice_to_tuple(&[], &process))
        );
    });
}

#[test]
fn with_list_returns_tuple() {
    with_process(|process| {
        let first_element = 1.into_process(&process);
        let second_element = 2.into_process(&process);
        let third_element = 3.into_process(&process);
        let list = Term::cons(
            first_element,
            Term::cons(
                second_element,
                Term::cons(third_element, Term::EMPTY_LIST, &process),
                &process,
            ),
            &process,
        );

        assert_eq!(
            erlang::list_to_tuple_1(list, &process),
            Ok(Term::slice_to_tuple(
                &[first_element, second_element, third_element],
                &process
            ))
        );
    });
}

#[test]
fn with_improper_list_errors_badarg() {
    errors_badarg(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_nested_list_returns_tuple_with_list_element() {
    with_process(|process| {
        // erlang doc: `[share, ['Ericsson_B', 163]]`
        let first_element = Term::str_to_atom("share", DoNotCare).unwrap();
        let second_element = Term::cons(
            Term::str_to_atom("Ericsson_B", DoNotCare).unwrap(),
            Term::cons(163.into_process(&process), Term::EMPTY_LIST, &process),
            &process,
        );
        let list = Term::cons(
            first_element,
            Term::cons(second_element, Term::EMPTY_LIST, &process),
            &process,
        );

        assert_eq!(
            erlang::list_to_tuple_1(list, &process),
            Ok(Term::slice_to_tuple(
                &[first_element, second_element],
                &process
            ))
        );
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
fn with_heap_binary_errors_badarg() {
    errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|process| bitstring!(1 :: 1, &process));
}

fn errors_badarg<L>(list: L)
where
    L: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        assert_badarg!(erlang::list_to_tuple_1(list(&process), &process));
    });
}
