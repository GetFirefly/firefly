use core::mem;

use crate::sys::dynamic_call::DynamicCallee;

pub unsafe fn apply(f: DynamicCallee, argv: *const usize, argc: usize) -> usize {
    match argc {
        0 => f(),
        1 => {
            let arity1 = mem::transmute::<_, extern "C" fn(usize) -> usize>(f);
            arity1(*argv)
        }
        2 => {
            let arity2 = mem::transmute::<_, extern "C" fn(usize, usize) -> usize>(f);
            arity2(*argv, *argv.offset(1))
        }
        3 => {
            let arity3 = mem::transmute::<_, extern "C" fn(usize, usize, usize) -> usize>(f);
            arity3(*argv, *argv.offset(1), *argv.offset(2))
        }
        4 => {
            let arity4 = mem::transmute::<_, extern "C" fn(usize, usize, usize, usize) -> usize>(f);
            arity4(*argv, *argv.offset(1), *argv.offset(2), *argv.offset(3))
        }
        5 => {
            let arity5 =
                mem::transmute::<_, extern "C" fn(usize, usize, usize, usize, usize) -> usize>(f);
            arity5(
                *argv,
                *argv.offset(1),
                *argv.offset(2),
                *argv.offset(3),
                *argv.offset(4),
            )
        }
        _ => unimplemented!("applying arity {} native functions", argc),
    }
}
