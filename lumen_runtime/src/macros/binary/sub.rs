#[cfg(test)]
macro_rules! bitstring {
    (@acc $bits:tt :: $bit_count:tt, $process:expr, $($byte:expr),*) => {{
       use crate::term::Term;

       let byte_count = <[()]>::len(&[$(replace_expr!($byte, ())),*]);
       let original = Term::slice_to_binary(&[$( $byte, )* $bits << (8 - $bit_count)], $process);
       Term::subbinary(original, 0, 0, byte_count, $bit_count, $process)
    }};
    (@acc $byte:expr, $($tail:tt)*) => {
       bitstring!(@acc $($tail)*, $byte)
    };
    ($bits:tt :: $bit_count:tt, $process:expr) => {
       bitstring!(@acc $bits :: $bit_count, $process,)
    };
    ($byte:expr, $($tail:tt)*) => {
       bitstring!(@acc $($tail)*, $byte)
    };
}

#[cfg(test)]
macro_rules! replace_expr {
    ($_t:expr, $sub:expr) => {
        $sub
    };
}
