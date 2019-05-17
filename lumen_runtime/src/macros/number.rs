macro_rules! number_infix_operator {
    ($left:ident, $right:ident, $process:ident, $checked:ident, $infix:tt) => {{
        use $crate::number::Operands::*;

        let operands = match ($left.tag(), $right.tag()) {
            (SmallInteger, SmallInteger) => {
                let left_isize = unsafe { $left.small_integer_to_isize() };
                let right_isize = unsafe { $right.small_integer_to_isize() };

                ISizes(left_isize, right_isize)
            }
            (SmallInteger, Boxed) => {
                let unboxed_right: &Term = $right.unbox_reference();

                match unboxed_right.tag() {
                    BigInteger => {
                        let left_big_int: BigInt = unsafe { $left.small_integer_to_big_int() };

                        let right_big_integer: &big::Integer = $right.unbox_reference();
                        let right_big_int = &right_big_integer.inner;

                        BigInts(left_big_int.clone(), right_big_int.clone())
                    }
                    Float => {
                        let left_isize = unsafe { $left.small_integer_to_isize() };
                        let left_f64: f64 = left_isize as f64;

                        let right_float: &Float = $right.unbox_reference();
                        let right_f64 = right_float.inner;

                        Floats(left_f64, right_f64)
                    }
                    _ => Bad
                }
            }
            (Boxed, SmallInteger) => {
                let unboxed_left: &Term = $left.unbox_reference();

                match unboxed_left.tag() {
                    BigInteger => {
                        let left_big_integer: &big::Integer = $left.unbox_reference();
                        let left_big_int = &left_big_integer.inner;

                        let right_big_int = unsafe { $right.small_integer_to_big_int() };

                        BigInts(left_big_int.clone(), right_big_int.clone())
                    }
                    Float => {
                        let left_float: &Float = $left.unbox_reference();
                        let left_f64 = left_float.inner;

                        let right_isize = unsafe { $right.small_integer_to_isize() };
                        let right_f64: f64 = right_isize as f64;

                        Floats(left_f64, right_f64)
                    }
                    _ => Bad
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

                        BigInts(left_big_int.clone(), right_big_int.clone())
                    }
                    (BigInteger, Float) => {
                        let left_big_integer: &big::Integer = $left.unbox_reference();
                        let left_f64: f64 = left_big_integer.into();

                        let right_float: &Float = $right.unbox_reference();
                        let right_f64 = right_float.inner;

                        Floats(left_f64, right_f64)
                    }
                    (Float, BigInteger) => {
                        let left_float: &Float = $left.unbox_reference();
                        let left_f64 = left_float.inner;

                        let right_big_intger: &big::Integer = $right.unbox_reference();
                        let right_f64: f64 = right_big_intger.into();

                        Floats(left_f64, right_f64)
                    }
                    (Float, Float) => {
                        let left_float: &Float = $left.unbox_reference();
                        let left_f64 = left_float.inner;

                        let right_float: &Float = $right.unbox_reference();
                        let right_f64 = right_float.inner;

                        Floats(left_f64, right_f64)
                    }
                    _ => Bad,
                }
            }
            _ => Bad
        };

        match operands {
            Bad => Err(badarith!()),
            ISizes(left_isize, right_isize) => {
                match left_isize.$checked(right_isize) {
                    Some(sum_isize) => Ok(sum_isize.into_process(&$process)),
                    None => {
                        let left_big_int: BigInt = left_isize.into();
                        let right_big_int: BigInt = right_isize.into();

                        let sum_big_int = left_big_int $infix right_big_int;

                        Ok(sum_big_int.into_process(&$process))
                    }
                }
            }
            Floats(left, right) => {
                let output = left $infix right;
                Ok(output.into_process($process))
            }
            BigInts(left, right) => {
                let output = left $infix right;
                Ok(output.into_process($process))
            }
        }
    }};
}
