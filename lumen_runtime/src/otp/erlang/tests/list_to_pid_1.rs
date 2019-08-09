use super::*;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(
                    erlang::list_to_pid_1(list, &arc_process),
                    Err(badarg!().into())
                );

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_list_encoding_local_pid() {
    with_process(|process| {
        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<").unwrap(),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<0").unwrap(),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<0.").unwrap(),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<0.1").unwrap(),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<0.1.").unwrap(),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<0.1.2").unwrap(),
            &process
        ));

        assert_eq!(
            erlang::list_to_pid_1(process.charlist_from_str("<0.1.2>").unwrap(), &process),
            Ok(make_pid(1, 2).unwrap())
        );

        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<0.1.2>?").unwrap(),
            &process
        ));
    })
}

#[test]
fn with_list_encoding_external_pid() {
    with_process(|process| {
        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<").unwrap(),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<1").unwrap(),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<1.").unwrap(),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<1.2").unwrap(),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<1.2.").unwrap(),
            &process
        ));
        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<1.2.3").unwrap(),
            &process
        ));

        assert_eq!(
            erlang::list_to_pid_1(process.charlist_from_str("<1.2.3>").unwrap(), &process),
            Ok(process.external_pid_with_node_id(1, 2, 3).unwrap())
        );

        assert_badarg!(erlang::list_to_pid_1(
            process.charlist_from_str("<1.2.3>?").unwrap(),
            &process
        ));
    });
}
