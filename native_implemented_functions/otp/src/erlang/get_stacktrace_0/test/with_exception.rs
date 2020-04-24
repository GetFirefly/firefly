use super::*;

use anyhow::*;

use liblumen_alloc::{atom, exit};

#[test]
fn without_stacktrace_returns_empty_list() {
    with_process(|process| {
        process.exception(exit!(atom!("reason"), anyhow!("Test").into()));

        assert_eq!(result(process), Term::NIL);
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

        process.exception(exit!(atom!("reason"), stacktrace, anyhow!("Test").into()));

        assert_eq!(result(process), stacktrace);
    })
}
