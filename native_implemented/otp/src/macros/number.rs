macro_rules! number_infix_operator {
    ($left:ident, $right:ident, $process:ident, $checked:ident, $infix:tt) => {{
        use anyhow::*;
        use num_bigint::BigInt;

        use firefly_rt::term::Term;

        use crate::number::Operands::*;

        let operands = match ($left, $right) {
            (Term::Int(left_small_integer), Term::Int(right_small_integer)) => {
                let left_isize = left_small_integer.into();
                let right_isize = right_small_integer.into();

                ISizes(left_isize, right_isize)
            }
            (Term::Int(left_small_integer), Term::BigInt(right_big_integer)) => {
                let left_big_int: BigInt = left_small_integer.into();
                let right_big_int: &BigInt = right_big_integer.as_ref().into();

                BigInts(left_big_int, right_big_int.clone())
            }
            (Term::Int(left_small_integer), Term::Float(right_float)) => {
                let left_f64: f64 = left_small_integer.into();
                let right_f64 = right_float.into();

                Floats(left_f64, right_f64)
            }
            (Term::BigInt(left_big_integer), Term::Int(right_small_integer)) => {
                let left_big_int: &BigInt = left_big_integer.as_ref().into();
                let right_big_int: BigInt = right_small_integer.into();

                BigInts(left_big_int.clone(), right_big_int)
            }
            (Term::Float(left_float), Term::Int(right_small_integer)) => {
                let left_f64 = left_float.into();
                let right_f64: f64 = right_small_integer.into();

                Floats(left_f64, right_f64)
            }
            (Term::BigInt(left_big_integer), Term::BigInt(right_big_integer)) => {
                let left_big_int: &BigInt = left_big_integer.as_ref().into();
                let right_big_int: &BigInt = right_big_integer.as_ref().into();

                BigInts(left_big_int.clone(), right_big_int.clone())
            }
            (Term::BigInt(left_big_integer), Term::Float(right_float)) => {
                let left_f64: f64 = left_big_integer.into();
                let right_f64 = right_float.into();

                Floats(left_f64, right_f64)
            }
            (Term::Float(left_float), Term::BigInt(right_big_integer)) => {
                let left_f64 = left_float.into();
                let right_f64: f64 = right_big_integer.into();

                Floats(left_f64, right_f64)
            }
            (Term::Float(left_float), Term::Float(right_float)) => {
                let left_f64 = left_float.into();
                let right_f64 = right_float.into();

                Floats(left_f64, right_f64)
            }
            _ => Bad
        };

        match operands {
            Bad => Err(
                badarith(
                    Trace::capture(),
                    Some(
                        anyhow!(
                            "{} ({}) and {} ({}) aren't both numbers",
                            stringify!($left),
                            $left,
                            stringify!($right),
                            $right
                        ).into()
                    )
                ).into()
            ),
            ISizes(left_isize, right_isize) => {
                match left_isize.$checked(right_isize) {
                    Some(sum_isize) => Ok($process.integer(sum_isize).unwrap()),
                    None => {
                        let left_big_int: BigInt = left_isize.into();
                        let right_big_int: BigInt = right_isize.into();

                        let sum_big_int = left_big_int $infix right_big_int;
                        let sum_term = $process.integer(sum_big_int).unwrap();

                        Ok(sum_term)
                    }
                }
            }
            Floats(left, right) => {
                let output = left $infix right;
                let output_term = output.into();

                Ok(output_term)
            }
            BigInts(left, right) => {
                let output = left $infix right;
                let output_term = $process.integer(output).unwrap();

                Ok(output_term)
            }
        }
    }};
}

macro_rules! number_to_integer {
    ($f:ident) => {
        use anyhow::*;
        use firefly_rt::error::ErlangException;
        use firefly_rt::process::Process;
        use firefly_rt::term::{Term, TypeError};

        use crate::erlang::number_to_integer::{f64_to_integer, NumberToInteger};

        #[native_implemented::function(erlang:$f/1)]
        pub fn result(process: &Process, number: Term) -> Result<Term, NonNull<ErlangException>> {
            match number.into() {
                NumberToInteger::Integer(integer) => Ok(integer),
                NumberToInteger::F64(f) => {
                    let ceiling = f.$f();

                    Ok(f64_to_integer(process, ceiling))
                }
                NumberToInteger::NotANumber => Err(TypeError)
                    .context(term_is_not_number!(number))
                    .map_err(From::from),
            }
        }
    };
}
