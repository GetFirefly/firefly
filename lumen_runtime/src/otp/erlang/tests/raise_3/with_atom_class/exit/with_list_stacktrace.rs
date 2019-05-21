use super::*;

#[test]
fn without_atom_module_errors_badarg() {
    with_stacktrace_errors_badarg(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_char_list("module", &process),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&process),
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn without_atom_function_errors_badarg() {
    with_stacktrace_errors_badarg(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_char_list("function", &process),
                    0.into_process(&process),
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn module_function_without_arity_or_arguments_errors_badarg() {
    with_stacktrace_errors_badarg(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    Term::str_to_atom("arity", DoNotCare).unwrap(),
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_module_function_arity_raises() {
    with_stacktrace_raises(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&process),
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_module_function_arguments_raises() {
    with_stacktrace_raises(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    Term::cons(
                        Term::str_to_atom("arg1", DoNotCare).unwrap(),
                        Term::EMPTY_LIST,
                        &process,
                    ),
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_mfa_with_empty_location_raises() {
    with_stacktrace_raises(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&process),
                    Term::EMPTY_LIST,
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_mfa_with_file_raises() {
    with_stacktrace_raises(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("file", DoNotCare).unwrap(),
                                Term::str_to_char_list("path_to_file", &process),
                            ],
                            &process,
                        ),
                        Term::EMPTY_LIST,
                        &process,
                    ),
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_mfa_with_file_without_char_list_errors_badarg() {
    with_stacktrace_errors_badarg(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("file", DoNotCare).unwrap(),
                                Term::slice_to_binary("path_to_file".as_bytes(), &process),
                            ],
                            &process,
                        ),
                        Term::EMPTY_LIST,
                        &process,
                    ),
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_mfa_with_line_raises() {
    with_stacktrace_raises(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("line", DoNotCare).unwrap(),
                                1.into_process(&process),
                            ],
                            &process,
                        ),
                        Term::EMPTY_LIST,
                        &process,
                    ),
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_mfa_with_line_with_zero_errors_badarg() {
    with_stacktrace_errors_badarg(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("line", DoNotCare).unwrap(),
                                0.into_process(&process),
                            ],
                            &process,
                        ),
                        Term::EMPTY_LIST,
                        &process,
                    ),
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_mfa_with_line_without_positive_integer_errors_badarg() {
    with_stacktrace_errors_badarg(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("line", DoNotCare).unwrap(),
                                Term::str_to_char_list("first", &process),
                            ],
                            &process,
                        ),
                        Term::EMPTY_LIST,
                        &process,
                    ),
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_mfa_with_file_and_line_raises() {
    with_stacktrace_raises(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("file", DoNotCare).unwrap(),
                                Term::str_to_char_list("path_to_file", &process),
                            ],
                            &process,
                        ),
                        Term::cons(
                            Term::slice_to_tuple(
                                &[
                                    Term::str_to_atom("line", DoNotCare).unwrap(),
                                    1.into_process(&process),
                                ],
                                &process,
                            ),
                            Term::EMPTY_LIST,
                            &process,
                        ),
                        &process,
                    ),
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

#[test]
fn with_mfa_with_invalid_location_errors_badarg() {
    with_stacktrace_errors_badarg(|process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("file", DoNotCare).unwrap(),
                                Term::str_to_char_list("path_to_file", &process),
                            ],
                            &process,
                        ),
                        Term::cons(
                            Term::slice_to_tuple(
                                &[
                                    // typo
                                    Term::str_to_atom("lin", DoNotCare).unwrap(),
                                    1.into_process(&process),
                                ],
                                &process,
                            ),
                            Term::EMPTY_LIST,
                            &process,
                        ),
                        &process,
                    ),
                ],
                &process,
            ),
            Term::EMPTY_LIST,
            &process,
        )
    })
}

fn with_stacktrace_raises<S>(stacktrace: S)
where
    S: FnOnce(&Process) -> Term,
{
    with(|class, reason, process| {
        let stacktrace = stacktrace(&process);

        assert_raises!(
            erlang::raise_3(class, reason, stacktrace),
            Exit,
            reason,
            Some(stacktrace)
        )
    })
}

fn with_stacktrace_errors_badarg<S>(stacktrace: S)
where
    S: FnOnce(&Process) -> Term,
{
    with(|class, reason, process| {
        assert_badarg!(erlang::raise_3(class, reason, stacktrace(&process)))
    })
}
