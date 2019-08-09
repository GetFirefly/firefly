use super::*;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(erlang::tl_1(list), Err(badarg!().into()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_errors_badarg() {
    assert_eq!(erlang::tl_1(Term::NIL), Err(badarg!().into()));
}

#[test]
fn with_list_returns_tail() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(head, tail)| {
                    let list = arc_process.cons(head, tail).unwrap();

                    prop_assert_eq!(erlang::tl_1(list), Ok(tail));

                    Ok(())
                },
            )
            .unwrap();
    });
}
