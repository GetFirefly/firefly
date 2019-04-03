use super::*;

#[test]
fn without_atom_module_errors_badarg() {
    with_stacktrace_errors_badarg(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_char_list("module", &mut process),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&mut process),
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn without_atom_function_errors_badarg() {
    with_stacktrace_errors_badarg(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_char_list("function", &mut process),
                    0.into_process(&mut process),
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn module_function_without_arity_or_arguments_errors_badarg() {
    with_stacktrace_errors_badarg(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    Term::str_to_atom("arity", DoNotCare).unwrap(),
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_module_function_arity_raises() {
    with_stacktrace_raises(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&mut process),
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_module_function_arguments_raises() {
    with_stacktrace_raises(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    Term::cons(
                        Term::str_to_atom("arg1", DoNotCare).unwrap(),
                        Term::EMPTY_LIST,
                        &mut process,
                    ),
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_mfa_with_empty_location_raises() {
    with_stacktrace_raises(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&mut process),
                    Term::EMPTY_LIST,
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_mfa_with_file_raises() {
    with_stacktrace_raises(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&mut process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("file", DoNotCare).unwrap(),
                                Term::str_to_char_list("path_to_file", &mut process),
                            ],
                            &mut process,
                        ),
                        Term::EMPTY_LIST,
                        &mut process,
                    ),
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_mfa_with_file_without_char_list_errors_badarg() {
    with_stacktrace_errors_badarg(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&mut process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("file", DoNotCare).unwrap(),
                                Term::slice_to_binary("path_to_file".as_bytes(), &mut process),
                            ],
                            &mut process,
                        ),
                        Term::EMPTY_LIST,
                        &mut process,
                    ),
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_mfa_with_line_raises() {
    with_stacktrace_raises(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&mut process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("line", DoNotCare).unwrap(),
                                1.into_process(&mut process),
                            ],
                            &mut process,
                        ),
                        Term::EMPTY_LIST,
                        &mut process,
                    ),
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_mfa_with_line_with_zero_errors_badarg() {
    with_stacktrace_errors_badarg(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&mut process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("line", DoNotCare).unwrap(),
                                0.into_process(&mut process),
                            ],
                            &mut process,
                        ),
                        Term::EMPTY_LIST,
                        &mut process,
                    ),
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_mfa_with_line_without_positive_integer_errors_badarg() {
    with_stacktrace_errors_badarg(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&mut process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("line", DoNotCare).unwrap(),
                                Term::str_to_char_list("first", &mut process),
                            ],
                            &mut process,
                        ),
                        Term::EMPTY_LIST,
                        &mut process,
                    ),
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_mfa_with_file_and_line_raises() {
    with_stacktrace_raises(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&mut process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("file", DoNotCare).unwrap(),
                                Term::str_to_char_list("path_to_file", &mut process),
                            ],
                            &mut process,
                        ),
                        Term::cons(
                            Term::slice_to_tuple(
                                &[
                                    Term::str_to_atom("line", DoNotCare).unwrap(),
                                    1.into_process(&mut process),
                                ],
                                &mut process,
                            ),
                            Term::EMPTY_LIST,
                            &mut process,
                        ),
                        &mut process,
                    ),
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

#[test]
fn with_mfa_with_invalid_location_errors_badarg() {
    with_stacktrace_errors_badarg(|mut process| {
        Term::cons(
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("module", DoNotCare).unwrap(),
                    Term::str_to_atom("function", DoNotCare).unwrap(),
                    0.into_process(&mut process),
                    Term::cons(
                        Term::slice_to_tuple(
                            &[
                                Term::str_to_atom("file", DoNotCare).unwrap(),
                                Term::str_to_char_list("path_to_file", &mut process),
                            ],
                            &mut process,
                        ),
                        Term::cons(
                            Term::slice_to_tuple(
                                &[
                                    // typo
                                    Term::str_to_atom("lin", DoNotCare).unwrap(),
                                    1.into_process(&mut process),
                                ],
                                &mut process,
                            ),
                            Term::EMPTY_LIST,
                            &mut process,
                        ),
                        &mut process,
                    ),
                ],
                &mut process,
            ),
            Term::EMPTY_LIST,
            &mut process,
        )
    })
}

fn with_stacktrace_raises<S>(stacktrace: S)
where
    S: FnOnce(&mut Process) -> Term,
{
    with(|class, reason, mut process| {
        let stacktrace = stacktrace(&mut process);

        assert_raises!(
            erlang::raise_3(class, reason, stacktrace),
            Error { arguments: None },
            reason,
            Some(stacktrace)
        )
    })
}

fn with_stacktrace_errors_badarg<S>(stacktrace: S)
where
    S: FnOnce(&mut Process) -> Term,
{
    with(|class, reason, mut process| {
        assert_badarg!(erlang::raise_3(class, reason, stacktrace(&mut process)))
    })
}
