use core::mem;

use crate::term::{ErlangResult, Term};

use super::DynamicCallee;

type DynamicCallee1 = extern "C-unwind" fn(Term) -> ErlangResult;
type DynamicCallee2 = extern "C-unwind" fn(Term, Term) -> ErlangResult;
type DynamicCallee3 = extern "C-unwind" fn(Term, Term, Term) -> ErlangResult;
type DynamicCallee4 = extern "C-unwind" fn(Term, Term, Term, Term) -> ErlangResult;
type DynamicCallee5 = extern "C-unwind" fn(Term, Term, Term, Term, Term) -> ErlangResult;

pub unsafe fn apply(f: DynamicCallee, argv: *const Term, argc: usize) -> ErlangResult {
    match argc {
        0 => f(),
        1 => {
            let arity1 = mem::transmute::<_, DynamicCallee1>(f);
            arity1(*argv)
        }
        2 => {
            let arity2 = mem::transmute::<_, DynamicCallee2>(f);
            arity2(*argv, *argv.offset(1))
        }
        3 => {
            let arity3 = mem::transmute::<_, DynamicCallee3>(f);
            arity3(*argv, *argv.offset(1), *argv.offset(2))
        }
        4 => {
            let arity4 = mem::transmute::<_, DynamicCallee4>(f);
            arity4(*argv, *argv.offset(1), *argv.offset(2), *argv.offset(3))
        }
        5 => {
            let arity5 = mem::transmute::<_, DynamicCallee5>(f);
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
