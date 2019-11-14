use super::*;

use std::convert::TryInto;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::{atom, badarg, badarith, badarity, badfun, badkey, badmap, exit};

use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_stacktrace_returns_empty_list() {
    with_process(|process| {
        process.exception(exit!(atom!("reason")));

        assert_eq!(native(process), Ok(Term::NIL));
    });
}

#[test]
fn with_stacktrace_returns_stacktrace() {
    with_process(|process| {
        let module = atom!("module");
        let function = atom!("function");
        let arity = 0.into();

        let file_key = atom!("file");
        let file_value = process.charlist_from_str("path.ex").unwrap();
        let file_tuple = process.tuple_from_slice(&[file_key, file_value]).unwrap();

        let line_key = atom!("line");
        let line_value = 1.into();
        let line_tuple = process.tuple_from_slice(&[line_key, line_value]).unwrap();

        let location = process.list_from_slice(&[file_tuple, line_tuple]).unwrap();

        let stack_item = process
            .tuple_from_slice(&[module, function, arity, location])
            .unwrap();

        let stacktrace = process.list_from_slice(&[stack_item]).unwrap();

        process.exception(exit!(atom!("reason"), stacktrace));

        assert_eq!(native(process), Ok(stacktrace));
    })
}

#[test]
fn badarg_includes_stacktrace() {
    with_process(|process| {
        process.exception(badarg!(process));

        assert_eq!(native(process), Ok(stacktrace(process)));
    })
}

#[test]
fn badarith_includes_stacktrace() {
    with_process(|process| {
        process.exception(badarith!(process));

        assert_eq!(native(process), Ok(stacktrace(process)));
    })
}

#[test]
fn badarity_includes_stacktrace() {
    with_process(|process| {
        let function = atom!("anonymous");
        let arguments = process.list_from_slice(&[]).unwrap();
        process.exception(badarity!(process, function, arguments).try_into().unwrap());

        assert_eq!(native(process), Ok(stacktrace(process)));
    })
}

#[test]
fn badfun_includes_stacktrace() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_function(arc_process.clone()),
                |function| {
                    arc_process.exception(badfun!(&arc_process, function).try_into().unwrap());

                    prop_assert_eq!(native(&arc_process), Ok(stacktrace(&arc_process)));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn badkey_includes_stacktrace() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |key| {
                arc_process.exception(badkey!(&arc_process, key).try_into().unwrap());

                prop_assert_eq!(native(&arc_process), Ok(stacktrace(&arc_process)));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn badmap_includes_stacktrace() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |map| {
                arc_process.exception(badmap!(&arc_process, map).try_into().unwrap());

                prop_assert_eq!(native(&arc_process), Ok(stacktrace(&arc_process)));

                Ok(())
            })
            .unwrap();
    });
}

fn stacktrace(process: &Process) -> Term {
    process
        .list_from_slice(&[process
            .tuple_from_slice(&[atom!("test"), atom!("loop"), 0.into()])
            .unwrap()])
        .unwrap()
}
