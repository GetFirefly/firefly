use super::*;

#[test]
fn without_proper_list_subtrahend_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_list(arc_process.clone()),
                    strategy::term::is_not_proper_list(arc_process.clone()),
                ),
                |(minuend, subtrahend)| {
                    prop_assert_eq!(
                        erlang::subtract_list_2(minuend, subtrahend, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_subtrahend_list_returns_minuend_with_first_copy_of_each_element_in_subtrahend_removed() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(element1, element2)| {
                    let mut heap = arc_process.acquire_heap();

                    let minuend = heap
                        .list_from_slice(&[element1, element2, element1])
                        .unwrap();
                    let subtrahend = heap.list_from_slice(&[element1]).unwrap();

                    prop_assert_eq!(
                        erlang::subtract_list_2(minuend, subtrahend, &arc_process),
                        Ok(heap.list_from_slice(&[element2, element1]).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
