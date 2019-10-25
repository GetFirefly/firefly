use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::Pid;

use crate::otp::erlang::list_to_pid_1::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(native(&arc_process, list), Err(badarg!().into()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_list_encoding_local_pid() {
    with_process(|process| {
        assert_badarg!(native(&process, process.charlist_from_str("<").unwrap()));
        assert_badarg!(native(&process, process.charlist_from_str("<0").unwrap()));
        assert_badarg!(native(&process, process.charlist_from_str("<0.").unwrap()));
        assert_badarg!(native(&process, process.charlist_from_str("<0.1").unwrap()));
        assert_badarg!(native(
            &process,
            process.charlist_from_str("<0.1.").unwrap(),
        ));
        assert_badarg!(native(
            &process,
            process.charlist_from_str("<0.1.2").unwrap(),
        ));

        assert_eq!(
            native(&process, process.charlist_from_str("<0.1.2>").unwrap()),
            Ok(Pid::make_term(1, 2).unwrap())
        );

        assert_badarg!(native(
            &process,
            process.charlist_from_str("<0.1.2>?").unwrap(),
        ));
    })
}

#[test]
fn with_list_encoding_external_pid() {
    with_process(|process| {
        assert_badarg!(native(&process, process.charlist_from_str("<").unwrap()));
        assert_badarg!(native(&process, process.charlist_from_str("<1").unwrap()));
        assert_badarg!(native(&process, process.charlist_from_str("<1.").unwrap()));
        assert_badarg!(native(&process, process.charlist_from_str("<1.2").unwrap()));
        assert_badarg!(native(
            &process,
            process.charlist_from_str("<1.2.").unwrap(),
        ));
        assert_badarg!(native(
            &process,
            process.charlist_from_str("<1.2.3").unwrap(),
        ));

        assert_eq!(
            native(&process, process.charlist_from_str("<1.2.3>").unwrap()),
            Ok(process.external_pid_with_node_id(1, 2, 3).unwrap())
        );

        assert_badarg!(native(
            &process,
            process.charlist_from_str("<1.2.3>?").unwrap(),
        ));
    });
}
