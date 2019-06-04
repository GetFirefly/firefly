use super::*;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(erlang::tl_1(list), Err(badarg!()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_errors_badarg() {
    assert_eq!(erlang::tl_1(Term::EMPTY_LIST), Err(badarg!()));
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
                    let list = Term::cons(head, tail, &arc_process);

                    prop_assert_eq!(erlang::tl_1(list), Ok(tail));

                    Ok(())
                },
            )
            .unwrap();
    });
}
