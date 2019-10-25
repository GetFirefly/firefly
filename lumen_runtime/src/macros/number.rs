macro_rules! number_infix_operator {
    ($left:ident, $right:ident, $process:ident, $checked:ident, $infix:tt) => {{
        use num_bigint::BigInt;

        use liblumen_alloc::badarith;
        use liblumen_alloc::erts::term::prelude::TypedTerm;

        use $crate::number::Operands::*;

        let operands = match ($left.decode().unwrap(), $right.decode().unwrap()) {
            (TypedTerm::SmallInteger(left_small_integer), TypedTerm::SmallInteger(right_small_integer)) => {
                let left_isize = left_small_integer.into();
                let right_isize = right_small_integer.into();

                ISizes(left_isize, right_isize)
            }
            (TypedTerm::SmallInteger(left_small_integer), TypedTerm::BigInteger(right_big_integer)) => {
                let left_big_int: BigInt = left_small_integer.into();
                let right_big_int: &BigInt = right_big_integer.as_ref().into();

                BigInts(left_big_int, right_big_int.clone())
            }
            (TypedTerm::SmallInteger(left_small_integer), TypedTerm::Float(right_float)) => {
                let left_f64: f64 = left_small_integer.into();
                let right_f64 = right_float.into();

                Floats(left_f64, right_f64)
            }
            (TypedTerm::BigInteger(left_big_integer), TypedTerm::SmallInteger(right_small_integer)) => {
                let left_big_int: &BigInt = left_big_integer.as_ref().into();
                let right_big_int: BigInt = right_small_integer.into();

                BigInts(left_big_int.clone(), right_big_int)
            }
            (TypedTerm::Float(left_float), TypedTerm::SmallInteger(right_small_integer)) => {
                let left_f64 = left_float.into();
                let right_f64: f64 = right_small_integer.into();

                Floats(left_f64, right_f64)
            }
            (TypedTerm::BigInteger(left_big_integer), TypedTerm::BigInteger(right_big_integer)) => {
                let left_big_int: &BigInt = left_big_integer.as_ref().into();
                let right_big_int: &BigInt = right_big_integer.as_ref().into();

                BigInts(left_big_int.clone(), right_big_int.clone())
            }
            (TypedTerm::BigInteger(left_big_integer), TypedTerm::Float(right_float)) => {
                let left_f64: f64 = left_big_integer.into();
                let right_f64 = right_float.into();

                Floats(left_f64, right_f64)
            }
            (TypedTerm::Float(left_float), TypedTerm::BigInteger(right_big_integer)) => {
                let left_f64 = left_float.into();
                let right_f64: f64 = right_big_integer.into();

                Floats(left_f64, right_f64)
            }
            (TypedTerm::Float(left_float), TypedTerm::Float(right_float)) => {
                let left_f64 = left_float.into();
                let right_f64 = right_float.into();

                Floats(left_f64, right_f64)
            }
            _ => Bad
        };

        match operands {
            Bad => Err(badarith!().into()),
            ISizes(left_isize, right_isize) => {
                match left_isize.$checked(right_isize) {
                    Some(sum_isize) => Ok($process.integer(sum_isize)?),
                    None => {
                        let left_big_int: BigInt = left_isize.into();
                        let right_big_int: BigInt = right_isize.into();

                        let sum_big_int = left_big_int $infix right_big_int;
                        let sum_term = $process.integer(sum_big_int)?;

                        Ok(sum_term)
                    }
                }
            }
            Floats(left, right) => {
                let output = left $infix right;
                let output_term = $process.float(output)?;

                Ok(output_term)
            }
            BigInts(left, right) => {
                let output = left $infix right;
                let output_term = $process.integer(output)?;

                Ok(output_term)
            }
        }
    }};
}
