#[cfg(all(not(target_arch = "wasm32"), test))]
macro_rules! bitstring {
    (@acc $bits:tt :: $bit_count:tt, $process:expr, $($byte:expr),*) => {{
        let byte_count = <[()]>::len(&[$(replace_expr!($byte, ())),*]);
        let original = $process.binary_from_bytes(&[$( $byte, )* $bits << (8 - $bit_count)]);

        $process.subbinary_from_original(original, 0, 0, byte_count, $bit_count)
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

#[cfg(all(not(target_arch = "wasm32"), test))]
macro_rules! replace_expr {
    ($_t:expr, $sub:expr) => {
        $sub
    };
}
