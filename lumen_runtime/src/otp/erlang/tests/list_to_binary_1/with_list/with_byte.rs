use super::*;

#[test]
fn without_byte_binary_or_list_element_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    byte(arc_process.clone()),
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
fn with_empty_list_returns_1_byte_binary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &any::<u8>().prop_map(|byte| {
                    (
                        Term::cons(
                            byte.into_process(&arc_process),
                            Term::EMPTY_LIST,
                            &arc_process,
                        ),
                        byte,
                    )
                }),
                |(list, byte)| {
                    let binary = Term::slice_to_binary(&[byte], &arc_process);

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
                &(byte(arc_process.clone()), byte(arc_process.clone()))
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
    with(|head_byte, head, process| {
        let tail_head_byte = 254;
        let tail_head = tail_head_byte.into_process(&process);

        let tail_tail = Term::EMPTY_LIST;

        let tail = Term::cons(tail_head, tail_tail, &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(Term::slice_to_binary(
                &[head_byte, tail_head_byte],
                &process
            ))
        );
    })
}

#[test]
fn with_heap_binary_returns_binary() {
    with(|head_byte, head, process| {
        let tail = Term::slice_to_binary(&[1, 2], &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(Term::slice_to_binary(&[head_byte, 1, 2], &process))
        );
    })
}

#[test]
fn with_subbinary_without_bitcount_returns_binary() {
    with(|head_byte, head, process| {
        let original = Term::slice_to_binary(&[0b0111_1111, 0b1000_0000], &process);
        let tail = Term::subbinary(original, 0, 1, 1, 0, &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(Term::slice_to_binary(&[head_byte, 255], &process))
        );
    })
}

#[test]
fn with_subbinary_with_bitcount_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    byte(arc_process.clone()),
                    strategy::term::binary::sub::is_not_binary(arc_process.clone()),
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

fn with<F>(f: F)
where
    F: FnOnce(u8, Term, &Process) -> (),
{
    with_process(|process| {
        let head_byte: u8 = 0;
        let head = head_byte.into_process(&process);

        f(head_byte, head, &process);
    })
}
