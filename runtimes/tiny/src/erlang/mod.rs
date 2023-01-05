pub mod file;
pub mod lists;
pub mod unicode;

use std::io::Write;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::Arc;

use smallvec::SmallVec;

use firefly_alloc::gc::GcBox;
use firefly_rt::backtrace::Trace;
use firefly_rt::error::ErlangException;
use firefly_rt::function::{self, ErlangResult, ModuleFunctionArity};
use firefly_rt::term::*;

use crate::scheduler;

macro_rules! handle_arith_result {
    ($math:expr) => {
        match $math {
            Ok(Number::Float(n)) => ErlangResult::Ok(n.into()),
            Ok(Number::Integer(n)) => handle_safe_integer_arith_result!(n),
            Err(_) => badarg(Trace::capture()),
        }
    };
}

macro_rules! handle_integer_arith_result {
    ($math:expr) => {
        match $math {
            Ok(result) => handle_safe_integer_arith_result!(result),
            Err(_) => badarg(Trace::capture()),
        }
    };
}

macro_rules! handle_safe_integer_arith_result {
    ($math:expr) => {
        match $math {
            Integer::Small(i) => ErlangResult::Ok(i.try_into().unwrap()),
            Integer::Big(i) => scheduler::with_current(|scheduler| {
                let arc_proc = scheduler.current_process();
                let proc = arc_proc.deref();

                let boxed = {
                    let mut empty = GcBox::new_uninit_in(proc).unwrap();
                    empty.write(i);
                    unsafe { empty.assume_init() }
                };
                ErlangResult::Ok(boxed.into())
            }),
        }
    };
}

#[export_name = "erlang:+/2"]
pub extern "C-unwind" fn plus2(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    handle_arith_result!(lhs + rhs)
}

#[export_name = "erlang:-/1"]
pub extern "C-unwind" fn neg1(lhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    handle_arith_result!(-lhs)
}

#[export_name = "erlang:-/2"]
pub extern "C-unwind" fn minus2(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    handle_arith_result!(lhs - rhs)
}

#[export_name = "erlang:*/2"]
pub extern "C-unwind" fn mul2(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    handle_arith_result!(lhs * rhs)
}

#[export_name = "erlang://2"]
pub extern "C-unwind" fn divide2(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    match lhs / rhs {
        Ok(result) => handle_arith_result!(result),
        Err(_) => badarg(Trace::capture()),
    }
}

#[export_name = "erlang:div/2"]
pub extern "C-unwind" fn div2(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    let lhs: Integer = lhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;
    let rhs: Integer = rhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;

    handle_integer_arith_result!(lhs / rhs)
}

#[export_name = "erlang:rem/2"]
pub extern "C-unwind" fn rem2(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    let lhs: Integer = lhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;
    let rhs: Integer = rhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;

    handle_integer_arith_result!(lhs % rhs)
}

#[export_name = "erlang:bsl/2"]
pub extern "C-unwind" fn bsl2(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    let lhs: Integer = lhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;
    let rhs: Integer = rhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;

    handle_integer_arith_result!(lhs << rhs)
}

#[export_name = "erlang:bsr/2"]
pub extern "C-unwind" fn bsr2(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    let lhs: Integer = lhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;
    let rhs: Integer = rhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;

    handle_integer_arith_result!(lhs >> rhs)
}

#[export_name = "erlang:band/2"]
pub extern "C-unwind" fn band2(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    let lhs: Integer = lhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;
    let rhs: Integer = rhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;

    handle_safe_integer_arith_result!(lhs & rhs)
}

#[export_name = "erlang:bnot/1"]
pub extern "C-unwind" fn bnot1(lhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    match lhs {
        Term::Bool(b) => ErlangResult::Ok((!b).into()),
        Term::Int(i) => {
            let i = !i;
            let term: Result<OpaqueTerm, _> = i.try_into();
            match term {
                Ok(t) => ErlangResult::Ok(t),
                Err(_) => scheduler::with_current(|scheduler| {
                    let arc_proc = scheduler.current_process();
                    let proc = arc_proc.deref();

                    let boxed = {
                        let mut empty = GcBox::new_uninit_in(proc).unwrap();
                        empty.write(BigInt::from(i));
                        unsafe { empty.assume_init() }
                    };
                    ErlangResult::Ok(boxed.into())
                }),
            }
        }
        Term::BigInt(i) => {
            let i = !(i.as_ref());
            scheduler::with_current(|scheduler| {
                let arc_proc = scheduler.current_process();
                let proc = arc_proc.deref();

                let boxed = {
                    let mut empty = GcBox::new_uninit_in(proc).unwrap();
                    empty.write(i);
                    unsafe { empty.assume_init() }
                };
                ErlangResult::Ok(boxed.into())
            })
        }
        _ => badarg(Trace::capture()),
    }
}

#[export_name = "erlang:bor/2"]
pub extern "C-unwind" fn bor2(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    let lhs: Integer = lhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;
    let rhs: Integer = rhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;

    handle_safe_integer_arith_result!(lhs | rhs)
}

#[export_name = "erlang:bxor/2"]
pub extern "C-unwind" fn bxor2(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    let lhs: Integer = lhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;
    let rhs: Integer = rhs.try_into().map_err(|_| badarg_err(Trace::capture()))?;

    handle_safe_integer_arith_result!(lhs ^ rhs)
}

#[export_name = "erlang:apply/2"]
pub extern "C-unwind" fn apply2(term: OpaqueTerm, arglist: OpaqueTerm) -> ErlangResult {
    let mut args = SmallVec::<[OpaqueTerm; 3]>::new();
    let callee = match (term.into(), arglist.into()) {
        (Term::Closure(fun), Term::Nil) => fun,
        (Term::Closure(fun), Term::Cons(ptr)) => {
            for element in unsafe { ptr.as_ref().iter().map(list_element_or_err) } {
                args.push(element?);
            }
            fun
        }
        _ => return badarg(Trace::capture()),
    };
    // Ensure the call is in tail position to allow for tail call optimization
    // if it can be applied by the compiler
    callee.apply(args.as_slice())
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:apply/3"]
pub extern "C-unwind" fn apply3(
    module: OpaqueTerm,
    function: OpaqueTerm,
    arglist: OpaqueTerm,
) -> ErlangResult {
    let mut args = SmallVec::<[OpaqueTerm; 3]>::new();
    let mfa = match (module.into(), function.into(), arglist.into()) {
        (Term::Atom(m), Term::Atom(f), Term::Nil) => ModuleFunctionArity::new(m, f, 0),
        (Term::Atom(m), Term::Atom(f), Term::Cons(ptr)) => {
            for element in unsafe { ptr.as_ref().iter().map(list_element_or_err) } {
                args.push(element?);
            }
            ModuleFunctionArity::new(m, f, args.len())
        }
        _ => return badarg(Trace::capture()),
    };
    let callee = match function::find_symbol(&mfa) {
        None => {
            let trace = Trace::capture();
            trace.set_top_frame(&mfa, args.as_slice());
            return undef(trace);
        }
        Some(callee) => callee,
    };
    // Ensure the call is in tail position to allow for tail call optimization
    // if it can be applied by the compiler
    unsafe { function::apply_callee(callee, args.as_slice()) }
}

#[track_caller]
fn list_element_or_err(element: Result<Term, ImproperList>) -> ErlangResult {
    match element {
        Ok(term) => ErlangResult::Ok(term.into()),
        Err(_) => {
            let exception = Box::into_raw(ErlangException::new(
                atoms::Error,
                atoms::Badarg.into(),
                Trace::capture(),
            ));
            ErlangResult::Err(unsafe { NonNull::new_unchecked(exception) })
        }
    }
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:make_fun/3"]
pub extern "C-unwind" fn make_fun3(
    module: OpaqueTerm,
    function: OpaqueTerm,
    arity: OpaqueTerm,
) -> ErlangResult {
    let Term::Atom(m) = module.into() else { panic!("invalid make_fun/3 bif module argument, expected atom, got: {:?}", module.r#typeof()); };
    let Term::Atom(f) = function.into() else { panic!("invalid make_fun/3 bif function argument, expected atom, got: {:?}", function.r#typeof()); };
    let Term::Int(a) = arity.into() else { panic!("invalid make_fun/3 bif arity argument, expected integer, got: {:?}", arity.r#typeof()); };

    let mfa = ModuleFunctionArity::new(m, f, a as usize);
    match function::find_symbol(&mfa) {
        Some(callee) => scheduler::with_current(|scheduler| {
            let arc_proc = scheduler.current_process();
            let proc = arc_proc.deref();

            ErlangResult::Ok(
                Closure::new_in(m, f, mfa.arity, callee as *const (), &[], proc)
                    .unwrap()
                    .into(),
            )
        }),
        None => undef(Trace::capture()),
    }
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:list_to_atom/1"]
pub extern "C-unwind" fn list_to_atom(term: OpaqueTerm) -> ErlangResult {
    match term.into() {
        Term::Nil => return ErlangResult::Ok(atoms::Empty.into()),
        Term::Cons(ptr) => {
            if let Some(s) = unsafe { ptr.as_ref().to_string() } {
                return ErlangResult::Ok(Atom::str_to_term(&s));
            }
        }
        _ => (),
    }
    let reason = make_reason(atoms::Badarg, term);
    raise2(reason, unsafe {
        NonNull::new_unchecked(Trace::into_raw(Trace::capture()))
    })
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:binary_to_list/1"]
pub extern "C-unwind" fn binary_to_list(term: OpaqueTerm) -> ErlangResult {
    let t: Term = term.into();
    if let Some(bits) = t.as_bitstring() {
        assert!(bits.is_binary());
        assert!(bits.is_aligned());

        scheduler::with_current(|scheduler| {
            let arc_proc = scheduler.current_process();
            let proc = arc_proc.deref();

            let bytes = unsafe { bits.as_bytes_unchecked() };
            let result = match core::str::from_utf8(bytes).ok() {
                Some(s) => Cons::charlist_from_str(s, proc).unwrap(),
                None => Cons::from_bytes(bytes, proc).unwrap(),
            };
            match result {
                None => ErlangResult::Ok(Term::Nil.into()),
                Some(cons) => ErlangResult::Ok(cons.into()),
            }
        })
    } else {
        let reason = make_reason(atoms::Badarg, term);
        raise2(reason, unsafe {
            NonNull::new_unchecked(Trace::into_raw(Trace::capture()))
        })
    }
}

#[export_name = "erlang:display/1"]
pub extern "C-unwind" fn display(term: OpaqueTerm) -> ErlangResult {
    let term: Term = term.into();
    println!("{}", &term);
    ErlangResult::Ok(true.into())
}

#[export_name = "erlang:debug/1"]
pub extern "C-unwind" fn debug(term: OpaqueTerm) -> ErlangResult {
    let term: Term = term.into();
    println!("{:?}", &term);
    ErlangResult::Ok(true.into())
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:display_nl/0"]
pub extern "C-unwind" fn display_nl() -> ErlangResult {
    println!();
    ErlangResult::Ok(true.into())
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:display_string/1"]
pub extern "C-unwind" fn display_string(term: OpaqueTerm) -> ErlangResult {
    let list: Term = term.into();
    match list {
        Term::Nil => ErlangResult::Ok(true.into()),
        Term::Cons(ptr) => {
            let cons = unsafe { ptr.as_ref() };
            match cons.to_string() {
                Some(ref s) => print!("{}", s),
                None => return badarg(Trace::capture()),
            }
            ErlangResult::Ok(true.into())
        }
        _other => badarg(Trace::capture()),
    }
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:puts/1"]
pub extern "C-unwind" fn puts(printable: OpaqueTerm) -> ErlangResult {
    let printable: Term = printable.into();

    let bits = printable.as_bitstring().unwrap();
    assert!(bits.is_aligned());
    assert!(bits.is_binary());
    let bytes = unsafe { bits.as_bytes_unchecked() };
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(bytes).unwrap();
    ErlangResult::Ok(true.into())
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:error/1"]
pub extern "C-unwind" fn error1(reason: OpaqueTerm) -> ErlangResult {
    let err = ErlangException::new(atoms::Error, reason.into(), Trace::capture());
    ErlangResult::Err(unsafe { NonNull::new_unchecked(Box::into_raw(err)) })
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:error/2"]
pub extern "C-unwind" fn error2(reason: OpaqueTerm, _args: OpaqueTerm) -> ErlangResult {
    error1(reason)
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:error/3"]
pub extern "C-unwind" fn error3(
    reason: OpaqueTerm,
    _args: OpaqueTerm,
    _options: OpaqueTerm,
) -> ErlangResult {
    error1(reason)
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:exit/1"]
pub extern "C-unwind" fn exit1(reason: OpaqueTerm) -> ErlangResult {
    let err = ErlangException::new(atoms::Exit, reason.into(), Trace::capture());
    ErlangResult::Err(unsafe { NonNull::new_unchecked(Box::into_raw(err)) })
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:throw/1"]
pub extern "C-unwind" fn throw1(reason: OpaqueTerm) -> ErlangResult {
    let err = ErlangException::new(atoms::Throw, reason.into(), Trace::capture());
    ErlangResult::Err(unsafe { NonNull::new_unchecked(Box::into_raw(err)) })
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:nif_error/1"]
pub extern "C-unwind" fn nif_error1(reason: OpaqueTerm) -> ErlangResult {
    error1(reason)
}

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:raise/2"]
pub extern "C-unwind" fn raise2(reason: OpaqueTerm, trace: NonNull<Trace>) -> ErlangResult {
    let trace = unsafe { Trace::from_raw(trace.as_ptr()) };
    let err = ErlangException::new(atoms::Error, reason.into(), trace);
    ErlangResult::Err(unsafe { NonNull::new_unchecked(Box::into_raw(err)) })
}

fn make_reason<R: Into<OpaqueTerm>>(tag: Atom, reason: R) -> OpaqueTerm {
    scheduler::with_current(|scheduler| {
        let arc_proc = scheduler.current_process();
        let proc = arc_proc.deref();
        Tuple::from_opaque_term_slice(&[tag.into(), reason.into()], proc)
            .unwrap()
            .into()
    })
}

pub(self) fn undef(trace: Arc<Trace>) -> ErlangResult {
    raise2(atoms::Undef.into(), unsafe {
        NonNull::new_unchecked(Trace::into_raw(trace))
    })
}

pub(self) fn badarg(trace: Arc<Trace>) -> ErlangResult {
    ErlangResult::Err(badarg_err(trace))
}

pub(self) fn badarg_err(trace: Arc<Trace>) -> NonNull<ErlangException> {
    let err = ErlangException::new(atoms::Error, atoms::Badarg.into(), trace);
    unsafe { NonNull::new_unchecked(Box::into_raw(err)) }
}
