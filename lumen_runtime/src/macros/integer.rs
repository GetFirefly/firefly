macro_rules! bitwise_infix_operator {
    ($left:ident, $right:ident, $process:ident, $infix:tt) => {{
        use num_bigint::BigInt;

        use liblumen_alloc::badarith;
        use liblumen_alloc::erts::term::prelude::TypedTerm;

        match ($left.to_typed_term().unwrap(), $right.to_typed_term().unwrap()) {
            (TypedTerm::SmallInteger(left_small_integer), TypedTerm::SmallInteger(right_small_integer)) => {
                let left_isize: isize = left_small_integer.into();
                let right_isize: isize = right_small_integer.into();
                let output = left_isize $infix right_isize;
                let output_term = $process.integer(output)?;

                Ok(output_term)
            }
            (TypedTerm::SmallInteger(left_small_integer), TypedTerm::BigInteger(right_big_integer)) => {
                let left_big_int: BigInt = left_small_integer.into();
                let right_big_int: &BigInt = right_big_integer.as_ref().into();

                let output_big_int = left_big_int $infix right_big_int;
                let output_term = $process.integer(output_big_int)?;

                Ok(output_term)
            }
            (TypedTerm::BigInteger(left_big_integer), TypedTerm::SmallInteger(right_small_integer)) => {
                let left_big_int: &BigInt = left_big_integer.as_ref().into();
                let right_big_int: BigInt = right_small_integer.into();

                let output_big_int = left_big_int $infix right_big_int;
                let output_term = $process.integer(output_big_int)?;

                Ok(output_term)
            }
            (TypedTerm::BigInteger(left_big_integer), TypedTerm::BigInteger(right_big_integer)) => {
                let left_big_int: &BigInt = left_big_integer.as_ref().into();
                let right_big_int: &BigInt = right_big_integer.as_ref().into();

                let output_big_int: BigInt = left_big_int $infix right_big_int;
                let output_term = $process.integer(output_big_int)?;

                Ok(output_term)
            }
            _ => Err(badarith!().into()),
        }
    }};
}

macro_rules! bitshift_infix_operator {
    ($integer:ident, $shift:ident, $process:ident, $positive:tt, $negative:tt) => {{
        use std::convert::TryInto;

        use num_bigint::BigInt;

        use liblumen_alloc::badarith;
        use liblumen_alloc::erts::term::prelude::TypedTerm;

        pub const MAX_SHIFT: usize = std::mem::size_of::<isize>() * 8 - 1;

        let option_shifted = match $integer.to_typed_term().unwrap() {
            TypedTerm::SmallInteger(integer_small_integer) => {
                let integer_isize: isize = integer_small_integer.into();
                let shift_isize: isize = $shift.try_into().map_err(|_| badarith!())?;

                // Rust doesn't support negative shift, so negative left shifts need to be right shifts
                if 0 <= shift_isize {
                    let shift_usize = shift_isize as usize;

                    if shift_usize <= MAX_SHIFT {
                        let shifted = integer_isize $positive shift_usize;
                        let shifted_term = $process.integer(shifted)?;

                        Some(shifted_term)
                    } else {
                        let big_int: BigInt = integer_isize.into();
                        let shifted = big_int $positive shift_usize;
                        let shifted_term = $process.integer(shifted)?;

                        Some(shifted_term)
                    }
                } else {
                    let shift_usize = (-shift_isize) as usize;

                    if shift_usize <= MAX_SHIFT {
                        let shifted = integer_isize $negative shift_usize;
                        let shifted_term = $process.integer(shifted)?;

                        Some(shifted_term)
                    } else {
                        let big_int: BigInt = integer_isize.into();
                        let shifted = big_int $negative shift_usize;
                        let shifted_term = $process.integer(shifted)?;

                        Some(shifted_term)
                    }
                }
            }
            TypedTerm::BigInteger(integer_big_integer) => {
                let big_int: &BigInt = integer_big_integer.as_ref().into();
                let shift_isize: isize = $shift.try_into().map_err(|_| badarith!())?;

                // Rust doesn't support negative shift, so negative left shifts need to be right
                // shifts
                let shifted = if 0 <= shift_isize {
                    let shift_usize = shift_isize as usize;

                    big_int $positive shift_usize
                } else {
                    let shift_usize = (-shift_isize) as usize;

                    big_int $negative shift_usize
                };

                let shifted_term = $process.integer(shifted)?;

                Some(shifted_term)
            }
            _ => None,
        };

        match option_shifted {
            Some(shifted) => Ok(shifted),
            None => Err(badarith!().into())
        }
    }};
}

macro_rules! integer_infix_operator {
    ($left:ident, $right:ident, $process:ident, $infix:tt) => {{
        use num_bigint::BigInt;

        use liblumen_alloc::badarith;
        use liblumen_alloc::erts::term::prelude::TypedTerm;

        match ($left.to_typed_term().unwrap(), $right.to_typed_term().unwrap()) {
            (TypedTerm::SmallInteger(left_small_integer), TypedTerm::SmallInteger(right_small_integer)) => {
                let left_isize: isize = left_small_integer.into();
                let right_isize: isize = right_small_integer.into();

                if right_isize == 0 {
                    Err(badarith!())
                } else {
                    let quotient = left_isize $infix right_isize;
                    let quotient_term = $process.integer(quotient)?;

                    Ok(quotient_term)
                }
            }
            (TypedTerm::SmallInteger(left_small_integre), TypedTerm::BigInteger(right_big_integer)) => {
                let left_big_int: BigInt = left_small_integre.into();
                let right_big_int: &BigInt = right_big_integer.as_ref().into();

                let quotient: BigInt = &left_big_int $infix right_big_int;
                let quotient_term = $process.integer(quotient)?;

                Ok(quotient_term)
            }
            (TypedTerm::BigInteger(left_big_integer), TypedTerm::SmallInteger(right_small_integer)) => {
                let right_isize: isize = right_small_integer.into();

                if right_isize == 0 {
                    Err(badarith!())
                } else {
                    let left_big_int: &BigInt = left_big_integer.as_ref().into();
                    let right_big_int: &BigInt = &right_isize.into();

                    let quotient = left_big_int $infix right_big_int;
                    let quotient_term = $process.integer(quotient)?;

                    Ok(quotient_term)
                }
            }
            (TypedTerm::BigInteger(left_big_integer), TypedTerm::BigInteger(right_big_integer)) => {
                let left_big_int: &BigInt = left_big_integer.as_ref().into();
                let right_big_int: &BigInt = right_big_integer.as_ref().into();

                let quotient = left_big_int $infix right_big_int;
                let quotient_term = $process.integer(quotient)?;

                Ok(quotient_term)
            }
            _ => Err(badarith!()),
        }.map_err(|error| error.into())
    }};
}
