use super::*;
use num_traits::Num;

#[test]
fn without_number_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_number(arc_process.clone()),
                |number| {
                    prop_assert_eq!(erlang::ceil_1(number, &arc_process), Err(badarg!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_returns_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_integer(arc_process.clone()), |number| {
                prop_assert_eq!(erlang::ceil_1(number, &arc_process), Ok(number));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_float_round_up_to_next_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::float(arc_process.clone()), |number| {
                let result = erlang::ceil_1(number, &arc_process);

                prop_assert!(result.is_ok());

                let result_term = result.unwrap();

                prop_assert!(result_term.is_integer());

                let number_f64: f64 = number.unbox_reference::<Float>().inner;

                if number_f64.fract() == 0.0 {
                    // f64::to_string() has no decimal point when there is no `fract`.
                    let number_big_int =
                        <BigInt as Num>::from_str_radix(&number_f64.to_string(), 10).unwrap();
                    let result_big_int: BigInt = result_term.try_into().unwrap();

                    prop_assert_eq!(number_big_int, result_big_int);
                } else {
                    prop_assert!(number <= result_term, "{:?} <= {:?}", number, result_term);
                }

                Ok(())
            })
            .unwrap();
    });
}
