#[cfg(all(not(target_arch = "wasm32"), test))]
macro_rules! bitstring {
    (@acc $bits:tt :: $bit_count:tt, $process:expr, $($byte:expr),*) => {{
        use liblumen_alloc::erts::process::alloc::TermAlloc;
        use liblumen_alloc::erts::term::prelude::Term;

        let byte_count = <[()]>::len(&[$(replace_expr!($byte, ())),*]);
        let mut heap = $process.acquire_heap();
        let original = heap.binary_from_bytes(&[$( $byte, )* $bits << (8 - $bit_count)]).unwrap();

        let term: Term = heap
            .subbinary_from_original(original, 0, 0, byte_count, $bit_count)
            .unwrap()
            .into();
        term
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
