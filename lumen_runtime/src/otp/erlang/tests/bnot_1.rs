use super::*;

use num_traits::Num;

#[test]
fn without_integer_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_integer(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(erlang::bnot_1(operand, &arc_process), Err(badarith!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_small_integer_returns_small_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::small(arc_process.clone()),
                |operand| {
                    let result = erlang::bnot_1(operand, &arc_process);

                    prop_assert!(result.is_ok());

                    let inverted = result.unwrap();

                    prop_assert_eq!(inverted.tag(), SmallInteger);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_small_integer_inverts_bits() {
    with_process(|process| {
        let integer = 0b10.into_process(&process);

        assert_eq!(
            erlang::bnot_1(integer, &process),
            Ok((-3).into_process(&process))
        );
    });
}

#[test]
fn with_big_integer_inverts_bits() {
    with_process(|process| {
        let integer = <BigInt as Num>::from_str_radix(
            "1010101010101010101010101010101010101010101010101010101010101010",
            2,
        )
        .unwrap()
        .into_process(&process);

        assert_eq!(integer.tag(), Boxed);

        let unboxed_integer: &Term = integer.unbox_reference();

        assert_eq!(unboxed_integer.tag(), BigInteger);

        assert_eq!(
            erlang::bnot_1(integer, &process),
            Ok(
                <BigInt as Num>::from_str_radix("-12297829382473034411", 10,)
                    .unwrap()
                    .into_process(&process)
            )
        );
    });
}

#[test]
fn with_big_integer_returns_big_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::big(arc_process.clone()),
                |operand| {
                    let result = erlang::bnot_1(operand, &arc_process);

                    prop_assert!(result.is_ok());

                    let inverted = result.unwrap();

                    prop_assert_eq!(inverted.tag(), Boxed);

                    let unboxed: &Term = inverted.unbox_reference();

                    prop_assert_eq!(unboxed.tag(), BigInteger);

                    Ok(())
                },
            )
            .unwrap();
    });
}
