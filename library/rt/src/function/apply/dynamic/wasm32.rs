use core::mem;

use crate::function::ErlangResult;
use crate::futures::ErlangFuture;
use crate::process::ProcessLock;
use crate::term::OpaqueTerm;

use super::DynamicCallee;

type DynamicCallee1 = extern "C-unwind" fn(&mut ProcessLock, OpaqueTerm) -> ErlangResult;
type DynamicCallee2 =
    extern "C-unwind" fn(&mut ProcessLock, OpaqueTerm, OpaqueTerm) -> ErlangResult;
type DynamicCallee3 =
    extern "C-unwind" fn(&mut ProcessLock, OpaqueTerm, OpaqueTerm, OpaqueTerm) -> ErlangResult;
type DynamicCallee4 = extern "C-unwind" fn(
    &mut ProcessLock,
    OpaqueTerm,
    OpaqueTerm,
    OpaqueTerm,
    OpaqueTerm,
) -> ErlangResult;
type DynamicCallee5 = extern "C-unwind" fn(
    &mut ProcessLock,
    OpaqueTerm,
    OpaqueTerm,
    OpaqueTerm,
    OpaqueTerm,
    OpaqueTerm,
) -> ErlangResult;

pub unsafe fn apply(
    f: DynamicCallee,
    process: &mut ProcessLock,
    argv: *const OpaqueTerm,
    argc: usize,
) -> ErlangResult {
    match argc {
        0 => f(process),
        1 => {
            let arity1 = mem::transmute::<_, DynamicCallee1>(f);
            arity1(process, *argv)
        }
        2 => {
            let arity2 = mem::transmute::<_, DynamicCallee2>(f);
            arity2(process, *argv, *argv.offset(1))
        }
        3 => {
            let arity3 = mem::transmute::<_, DynamicCallee3>(f);
            arity3(process, *argv, *argv.offset(1), *argv.offset(2))
        }
        4 => {
            let arity4 = mem::transmute::<_, DynamicCallee4>(f);
            arity4(
                process,
                *argv,
                *argv.offset(1),
                *argv.offset(2),
                *argv.offset(3),
            )
        }
        5 => {
            let arity5 = mem::transmute::<_, DynamicCallee5>(f);
            arity5(
                process,
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

#[cfg(feature = "async")]
mod async_impl {
    use super::super::DynamicAsyncCallee;

    type DynamicAsyncCallee1 = extern "C-unwind" fn(&mut ProcessLock, OpaqueTerm) -> ErlangFuture;
    type DynamicAsyncCallee2 =
        extern "C-unwind" fn(&mut ProcessLock, OpaqueTerm, OpaqueTerm) -> ErlangFuture;
    type DynamicAsyncCallee3 =
        extern "C-unwind" fn(&mut ProcessLock, OpaqueTerm, OpaqueTerm, OpaqueTerm) -> ErlangFuture;
    type DynamicAsyncCallee4 = extern "C-unwind" fn(
        &mut ProcessLock,
        OpaqueTerm,
        OpaqueTerm,
        OpaqueTerm,
        OpaqueTerm,
    ) -> ErlangFuture;
    type DynamicAsyncCallee5 = extern "C-unwind" fn(
        &mut ProcessLock,
        OpaqueTerm,
        OpaqueTerm,
        OpaqueTerm,
        OpaqueTerm,
        OpaqueTerm,
    ) -> ErlangFuture;

    pub unsafe fn apply_async(
        f: DynamicAsyncCallee,
        process: &mut ProcessLock,
        argv: *const OpaqueTerm,
        argc: usize,
    ) -> ErlangFuture {
        match argc {
            0 => f(process),
            1 => {
                let arity1 = mem::transmute::<_, DynamicAsyncCallee1>(f);
                arity1(process, *argv)
            }
            2 => {
                let arity2 = mem::transmute::<_, DynamicAsyncCallee2>(f);
                arity2(process, *argv, *argv.offset(1))
            }
            3 => {
                let arity3 = mem::transmute::<_, DynamicAsyncCallee3>(f);
                arity3(process, *argv, *argv.offset(1), *argv.offset(2))
            }
            4 => {
                let arity4 = mem::transmute::<_, DynamicAsyncCallee4>(f);
                arity4(
                    process,
                    *argv,
                    *argv.offset(1),
                    *argv.offset(2),
                    *argv.offset(3),
                )
            }
            5 => {
                let arity5 = mem::transmute::<_, DynamicAsyncCallee5>(f);
                arity5(
                    process,
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
}

#[cfg(feature = "async")]
pub use self::async_impl::apply_async;
