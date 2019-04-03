macro_rules! infix_operator {
    ($left:ident, $right:ident, $process:ident, $checked:ident, $infix:tt) => {
        match ($left.tag(), $right.tag()) {
            (SmallInteger, SmallInteger) => {
                let left_isize = unsafe { $left.small_integer_to_isize() };
                let right_isize = unsafe { $right.small_integer_to_isize() };

                match left_isize.$checked(right_isize) {
                    Some(sum_isize) => Ok(sum_isize.into_process(&mut $process)),
                    None => {
                        let left_big_int: BigInt = left_isize.into();
                        let right_big_int: BigInt = right_isize.into();

                        let sum_big_int = left_big_int $infix right_big_int;

                        Ok(sum_big_int.into_process(&mut $process))
                    }
                }
            }
            (SmallInteger, Boxed) => {
                let unboxed_right: &Term = $right.unbox_reference();

                match unboxed_right.tag() {
                    BigInteger => {
                        let left_isize = unsafe { $left.small_integer_to_isize() };
                        let left_big_int: BigInt = left_isize.into();

                        let right_big_integer: &big::Integer = $right.unbox_reference();
                        let right_big_int = &right_big_integer.inner;

                        let sum_big_int = left_big_int $infix right_big_int;

                        Ok(sum_big_int.into_process(&mut $process))
                    }
                    Float => {
                        let left_isize = unsafe { $left.small_integer_to_isize() };
                        let left_f64: f64 = left_isize as f64;

                        let right_float: &Float = $right.unbox_reference();
                        let right_f64 = right_float.inner;

                        let sum_f64 = left_f64 $infix right_f64;

                        Ok(sum_f64.into_process(&mut $process))
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

                        let sum_big_int = left_big_int $infix right_big_int;

                        Ok(sum_big_int.into_process(&mut $process))
                    }
                    Float => {
                        let left_float: &Float = $left.unbox_reference();
                        let left_f64 = left_float.inner;

                        let right_isize = unsafe { $right.small_integer_to_isize() };
                        let right_f64: f64 = right_isize as f64;

                        let sum_f64 = left_f64 $infix right_f64;

                        Ok(sum_f64.into_process(&mut $process))
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

                        let sum_big_int = left_big_int $infix right_big_int;

                        Ok(sum_big_int.into_process(&mut $process))
                    }
                    (BigInteger, Float) => {
                        let left_big_integer: &big::Integer = $left.unbox_reference();
                        let left_f64: f64 = left_big_integer.into();

                        let right_float: &Float = $right.unbox_reference();
                        let right_f64 = right_float.inner;

                        let sum_f64 = left_f64 $infix right_f64;

                        Ok(sum_f64.into_process(&mut $process))
                    }
                    (Float, BigInteger) => {
                        let left_float: &Float = $left.unbox_reference();
                        let left_f64 = left_float.inner;

                        let right_big_intger: &big::Integer = $right.unbox_reference();
                        let right_f64: f64 = right_big_intger.into();

                        let sum_f64 = left_f64 $infix right_f64;

                        Ok(sum_f64.into_process(&mut $process))
                    }
                    (Float, Float) => {
                        let left_float: &Float = $left.unbox_reference();
                        let left_f64 = left_float.inner;

                        let right_float: &Float = $right.unbox_reference();
                        let right_f64 = right_float.inner;

                        let sum_f64 = left_f64 $infix right_f64;

                        Ok(sum_f64.into_process(&mut $process))
                    }
                    _ => Err(badarith!()),
                }
            }
            _ => Err(badarith!()),
        }
    };
}
