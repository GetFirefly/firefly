macro_rules! bitwise_infix_operator {
    ($left:ident, $right:ident, $process:ident, $infix:tt) => {
        match ($left.tag(), $right.tag()) {
            (SmallInteger, SmallInteger) => {
                let left_isize = unsafe { $left.small_integer_to_isize() };
                let right_isize = unsafe { $right.small_integer_to_isize() };
                let output = left_isize $infix right_isize;

                Ok(output.into_process(&$process))
            }
            (SmallInteger, Boxed) => {
                let unboxed_right: &Term = $right.unbox_reference();

                match unboxed_right.tag() {
                    BigInteger => {
                        let left_isize = unsafe { $left.small_integer_to_isize() };
                        let left_big_int: BigInt = left_isize.into();

                        let right_big_integer: &big::Integer = $right.unbox_reference();
                        let right_big_int = &right_big_integer.inner;

                        let output_big_int = left_big_int $infix right_big_int;

                        Ok(output_big_int.into_process(&$process))
                    }
                    _ => Err(badarith!()),
                }
            }
            (Boxed, SmallInteger) => {
                let unboxed_left: &Term = $left.unbox_reference();

                match unboxed_left.tag() {
                    BigInteger => {
                        let left_big_integer: &big::Integer = $left.unbox_reference();
                        let left_big_int = &left_big_integer.inner;

                        let right_isize = unsafe { $right.small_integer_to_isize() };
                        let right_big_int: BigInt = right_isize.into();

                        let output_big_int = left_big_int $infix right_big_int;

                        Ok(output_big_int.into_process(&$process))
                    }
                    _ => Err(badarith!()),
                }
            }
            (Boxed, Boxed) => {
                let unboxed_left: &Term = $left.unbox_reference();
                let unboxed_right: &Term = $right.unbox_reference();

                match (unboxed_left.tag(), unboxed_right.tag()) {
                    (BigInteger, BigInteger) => {
                        let left_big_integer: &big::Integer = $left.unbox_reference();
                        let left_big_int = &left_big_integer.inner;

                        let right_big_integer: &big::Integer = $right.unbox_reference();
                        let right_big_int = &right_big_integer.inner;

                        let output_big_int = left_big_int $infix right_big_int;

                        Ok(output_big_int.into_process(&$process))
                    }
                    _ => Err(badarith!()),
                }
            }
            _ => Err(badarith!()),
        }
    };
}

macro_rules! bitshift_infix_operator {
    ($integer:ident, $shift:ident, $process:ident, $positive:tt, $negative:tt) => {{
        pub const MAX_SHIFT: usize = std::mem::size_of::<isize>() * 8 - 1;

        match $integer.tag() {
            SmallInteger => {
                let integer_isize: isize = $integer.try_into()?;
                let shift_isize: isize = $shift.try_into().map_err(|_| badarith!())?;

                // Rust doesn't support negative shift, so negative left shifts need to be right shifts
                if 0 <= shift_isize {
                    let shift_usize = shift_isize as usize;

                    if shift_usize <= MAX_SHIFT {
                        let shifted = integer_isize $positive shift_usize;

                        Ok(shifted.into_process(&$process))
                    } else {
                        let big_int: BigInt = integer_isize.into();
                        let shifted = big_int $positive shift_usize;

                        Ok(shifted.into_process(&$process))
                    }
                } else {
                    let shift_usize = (-shift_isize) as usize;

                    if shift_usize <= MAX_SHIFT {
                        let shifted = integer_isize $negative shift_usize;

                        Ok(shifted.into_process(&$process))
                    } else {
                        let big_int: BigInt = integer_isize.into();
                        let shifted = big_int $negative shift_usize;

                        Ok(shifted.into_process(&$process))
                    }
                }
            }
            Boxed => {
                let unboxed_integer: &Term = $integer.unbox_reference();

                match unboxed_integer.tag() {
                    BigInteger => {
                        let big_integer: &big::Integer = $integer.unbox_reference();
                        let big_int: &BigInt = &big_integer.inner;
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

                        Ok(shifted.into_process(&$process))
                    }
                    _ => Err(badarith!()),
                }
            }
            _ => Err(badarith!()),
        }
    }};
}

macro_rules! integer_infix_operator {
    ($left:ident, $right:ident, $process:ident, $infix:tt) => {
        match ($left.tag(), $right.tag()) {
            (SmallInteger, SmallInteger) => {
                let left_isize = unsafe { $left.small_integer_to_isize() };
                let right_isize = unsafe { $right.small_integer_to_isize() };

                if right_isize == 0 {
                    Err(badarith!())
                } else {
                    let quotient = left_isize $infix right_isize;

                    Ok(quotient.into_process(&$process))
                }
            }
            (SmallInteger, Boxed) => {
                let right_unboxed: &Term = $right.unbox_reference();

                match right_unboxed.tag() {
                    BigInteger => {
                        let left_isize = unsafe { $left.small_integer_to_isize() };
                        let left_big_int: &BigInt = &left_isize.into();

                        let right_big_integer: &big::Integer = $right.unbox_reference();
                        let right_big_int = &right_big_integer.inner;

                        let quotient = left_big_int $infix right_big_int;

                        Ok(quotient.into_process(&$process))
                    }
                    _ => Err(badarith!()),
                }
            }
            (Boxed, SmallInteger) => {
                let left_unboxed: &Term = $left.unbox_reference();

                match left_unboxed.tag() {
                    BigInteger => {
                        let left_big_integer: &big::Integer = $left.unbox_reference();
                        let left_big_int = &left_big_integer.inner;

                        let right_isize = unsafe { $right.small_integer_to_isize() };

                        if right_isize == 0 {
                            Err(badarith!())
                        } else {
                            let right_big_int: &BigInt = &right_isize.into();

                            let quotient = left_big_int $infix right_big_int;

                            Ok(quotient.into_process(&$process))
                        }
                    }
                    _ => Err(badarith!()),
                }
            }
            (Boxed, Boxed) => {
                let left_unboxed: &Term = $left.unbox_reference();
                let right_unboxed: &Term = $right.unbox_reference();

                match (left_unboxed.tag(), right_unboxed.tag()) {
                    (BigInteger, BigInteger) => {
                        let left_big_integer: &big::Integer = $left.unbox_reference();
                        let left_big_int = &left_big_integer.inner;

                        let right_big_integer: &big::Integer = $right.unbox_reference();
                        let right_big_int = &right_big_integer.inner;

                        let quotient = left_big_int $infix right_big_int;

                        Ok(quotient.into_process(&$process))
                    }
                    _ => Err(badarith!()),
                }
            }
            _ => Err(badarith!()),
        }
    };
}
