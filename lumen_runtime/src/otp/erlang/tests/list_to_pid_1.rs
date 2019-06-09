use super::*;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(erlang::list_to_pid_1(list, &arc_process), Err(badarg!()));

                Ok(())
            })
            .unwrap();
    });
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
