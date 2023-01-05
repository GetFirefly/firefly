use firefly_rt::term::Atom;
use super::*;

#[test]
fn without_stacktrace_returns_empty_list() {
    with_process(|process| {
        process.exception(exit!(
            atom!("reason"),
            Trace::capture(),
            anyhow!("Test").into()
        ));

        assert_eq!(result(process), Term::Nil);
    });
}

#[test]
fn with_stacktrace_returns_stacktrace() {
    with_process(|process| {
        let module = Atom::str_to_term("module").into();
        let function = Atom::str_to_term("function").into();
        let arity = 0.into();

        let file_key = atoms::File.into();
        let file_value = process.charlist_from_str("path.ex");
        let file_tuple = process.tuple_term_from_term_slice(&[file_key, file_value]);

        let line_key = atoms::Line.into();
        let line_value = 1.into();
        let line_tuple = process.tuple_term_from_term_slice(&[line_key, line_value]);

        let location = Term::list_from_slice_in(&[file_tuple, line_tuple], proces);

        let stack_item = process.tuple_term_from_term_slice(&[module, function, arity, location]);

        let stacktrace = process.list_from_slice(&[stack_item]);

        let arc_trace = Trace::from_term(stacktrace);
        process.exception(exit!(Atom::str_to_term("reason").into(), arc_trace));

        assert_eq!(result(process), stacktrace);
    })
}
