use super::*;

#[test]
fn without_byte_binary_or_list_element_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::binary::heap(arc_process.clone()),
                    is_not_byte_binary_nor_list(arc_process.clone()),
                )
                    .prop_map(|(head, tail)| Term::cons(head, tail, &arc_process)),
                |list| {
                    prop_assert_eq!(erlang::list_to_binary_1(list, &arc_process), Err(badarg!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_list_returns_binary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::heap(arc_process.clone()).prop_map(|binary| {
                    (Term::cons(binary, Term::EMPTY_LIST, &arc_process), binary)
                }),
                |(list, binary)| {
                    prop_assert_eq!(erlang::list_to_binary_1(list, &arc_process), Ok(binary));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_byte_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::binary::heap(arc_process.clone()),
                    byte(arc_process.clone()),
                )
                    .prop_map(|(head, tail)| Term::cons(head, tail, &arc_process)),
                |list| {
                    prop_assert_eq!(erlang::list_to_binary_1(list, &arc_process), Err(badarg!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_list_without_byte_tail_returns_binary() {
    with(|head, process| {
        let tail_head_byte = 3;
        let tail_head = tail_head_byte.into_process(&process);

        let tail_tail = Term::EMPTY_LIST;

        let tail = Term::cons(tail_head, tail_tail, &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(Term::slice_to_binary(&[0, 1, tail_head_byte], &process))
        );
    })
}

#[test]
fn with_heap_binary_returns_binary() {
    with(|head, process| {
        let tail = Term::slice_to_binary(&[2, 3], &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(Term::slice_to_binary(&[0, 1, 2, 3], &process))
        );
    })
}

#[test]
fn with_subbinary_without_bitcount_returns_binary() {
    with(|head, process| {
        let original = Term::slice_to_binary(&[0b0111_1111, 0b1000_0000], &process);
        let tail = Term::subbinary(original, 0, 1, 1, 0, &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(Term::slice_to_binary(&[0, 1, 255], &process))
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let head = Term::slice_to_binary(&[0, 1], &process);

        f(head, &process);
    })
}
