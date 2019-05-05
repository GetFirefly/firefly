use super::*;

use crate::process::IntoProcess;

#[test]
fn with_atom_errors_with_atom_reason() {
    with_process(|process| {
        let reason = Term::str_to_atom("reason", DoNotCare).unwrap();
        let arguments = Term::cons(
            Term::str_to_atom("first", DoNotCare).unwrap(),
            Term::EMPTY_LIST,
            &process,
        );

        assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
    });
}

#[test]
fn with_list_reference_errors_with_list_reference_reason() {
    with_process(|process| {
        let reason = Term::next_local_reference(process);
        let arguments = Term::EMPTY_LIST;

        assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
    });
}

#[test]
fn with_empty_list_errors_with_empty_list_reason() {
    let reason = Term::EMPTY_LIST;
    let arguments = Term::EMPTY_LIST;

    assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
}

#[test]
fn with_list_errors_with_list_reason() {
    with_process(|process| {
        let reason = list_term(&process);
        let arguments = Term::cons(list_term(&process), Term::EMPTY_LIST, &process);

        assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
    });
}

#[test]
fn with_small_integer_errors_with_small_integer_reason() {
    with_process(|process| {
        let reason = 0usize.into_process(&process);
        let arguments = Term::cons(1.into_process(&process), Term::EMPTY_LIST, &process);

        assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
    });
}

#[test]
fn with_big_integer_errors_with_big_integer_reason() {
    with_process(|process| {
        let reason = (integer::small::MAX + 1).into_process(&process);
        let arguments = Term::cons(
            (integer::small::MAX + 1).into_process(&process),
            Term::EMPTY_LIST,
            &process,
        );

        assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
    });
}

#[test]
fn with_float_errors_with_float_reason() {
    with_process(|process| {
        let reason = 1.0.into_process(&process);
        let arguments = Term::cons(2.0.into_process(&process), Term::EMPTY_LIST, &process);

        assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
    });
}

#[test]
fn with_local_pid_errors_with_local_pid_reason() {
    with_process(|process| {
        let reason = Term::local_pid(0, 0).unwrap();
        let arguments = Term::cons(Term::local_pid(1, 2).unwrap(), Term::EMPTY_LIST, &process);

        assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
    });
}

#[test]
fn with_external_pid_errors_with_external_pid_reason() {
    with_process(|process| {
        let reason = Term::external_pid(1, 0, 0, &process).unwrap();
        let arguments = Term::cons(
            Term::external_pid(2, 3, 4, &process).unwrap(),
            Term::EMPTY_LIST,
            &process,
        );

        assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
    });
}

#[test]
fn with_tuple_errors_with_tuple_reason() {
    with_process(|process| {
        let reason = Term::slice_to_tuple(&[], &process);
        let arguments = Term::cons(
            Term::slice_to_tuple(&[1.into_process(&process)], &process),
            Term::EMPTY_LIST,
            &process,
        );

        assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
    });
}

#[test]
fn with_map_errors_with_map_reason() {
    with_process(|process| {
        let reason = Term::slice_to_map(
            &[(
                Term::str_to_atom("a", DoNotCare).unwrap(),
                1.into_process(&process),
            )],
            &process,
        );
        let arguments = Term::cons(
            Term::slice_to_map(
                &[(
                    Term::str_to_atom("b", DoNotCare).unwrap(),
                    2.into_process(&process),
                )],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        );

        assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
    });
}

#[test]
fn with_heap_binary_errors_with_heap_binary_reason() {
    with_process(|process| {
        let reason = Term::slice_to_binary(&[0], &process);
        let arguments = Term::cons(
            Term::slice_to_binary(&[1], &process),
            Term::EMPTY_LIST,
            &process,
        );

        assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
    });
}

#[test]
fn with_subbinary_errors_with_subbinary_reason() {
    with_process(|process| {
        // <<1::1, 2>>
        let reason_original = Term::slice_to_binary(&[129, 0b0000_0000], &process);
        let reason = Term::subbinary(reason_original, 0, 1, 1, 0, &process);

        // <<3::3, 4>>
        let argument_original = Term::slice_to_binary(&[96, 0b0100_0000], &process);
        let argument = Term::subbinary(argument_original, 0, 2, 1, 0, &process);
        let arguments = Term::cons(argument, Term::EMPTY_LIST, &process);

        assert_error!(erlang::error_2(reason, arguments), reason, Some(arguments));
    });
}
