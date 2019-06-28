use super::*;

#[test]
fn without_number_addend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::small(arc_process.clone()),
                    strategy::term::is_not_number(arc_process.clone()),
                ),
                |(augend, addend)| {
                    prop_assert_eq!(
                        erlang::add_2(augend, addend, &arc_process),
                        Err(badarith!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_small_integer_addend_without_underflow_or_overflow_returns_small_integer() {
    with(|augend, process| {
        let addend = 3.into_process(&process);

        assert_eq!(
            erlang::add_2(augend, addend, &process),
            Ok(5.into_process(&process))
        );
    })
}

#[test]
fn with_small_integer_addend_with_underflow_returns_big_integer() {
    with_process(|process| {
        let augend = (-1_isize).into_process(&process);
        let addend = crate::integer::small::MIN.into_process(&process);

        assert_eq!(addend.tag(), SmallInteger);

        let result = erlang::add_2(augend, addend, &process);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert_eq!(sum.tag(), Boxed);

        let unboxed_sum: &Term = sum.unbox_reference();

        assert_eq!(unboxed_sum.tag(), BigInteger);
    })
}

#[test]
fn with_small_integer_addend_with_overflow_returns_big_integer() {
    with(|augend, process| {
        let addend = crate::integer::small::MAX.into_process(&process);

        assert_eq!(addend.tag(), SmallInteger);

        let result = erlang::add_2(augend, addend, &process);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert_eq!(sum.tag(), Boxed);

        let unboxed_sum: &Term = sum.unbox_reference();

        assert_eq!(unboxed_sum.tag(), BigInteger);
    })
}

#[test]
fn with_big_integer_addend_returns_big_integer() {
    with(|augend, process| {
        let addend = (crate::integer::small::MAX + 1).into_process(&process);

        assert_eq!(addend.tag(), Boxed);

        let unboxed_addend: &Term = addend.unbox_reference();

        assert_eq!(unboxed_addend.tag(), BigInteger);

        let result = erlang::add_2(augend, addend, &process);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert_eq!(sum.tag(), Boxed);

        let unboxed_sum: &Term = sum.unbox_reference();

        assert_eq!(unboxed_sum.tag(), BigInteger);
    })
}

#[test]
fn with_float_addend_without_underflow_or_overflow_returns_float() {
    with(|augend, process| {
        let addend = 3.0.into_process(&process);

        assert_eq!(
            erlang::add_2(augend, addend, &process),
            Ok(5.0.into_process(&process))
        );
    })
}

#[test]
fn with_float_addend_with_underflow_returns_min_float() {
    with(|augend, process| {
        let addend = std::f64::MIN.into_process(&process);

        assert_eq!(
            erlang::add_2(augend, addend, &process),
            Ok(std::f64::MIN.into_process(&process))
        );
    })
}

#[test]
fn with_float_addend_with_overflow_returns_max_float() {
    with(|augend, process| {
        let addend = std::f64::MAX.into_process(&process);

        assert_eq!(
            erlang::add_2(augend, addend, &process),
            Ok(std::f64::MAX.into_process(&process))
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let augend = 2.into_process(&process);

        f(augend, &process)
    })
}
