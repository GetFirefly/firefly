pub mod file;
pub mod lists;
pub mod unicode;

use std::io::Write;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::Arc;

use smallvec::SmallVec;

use liblumen_rt::backtrace::Trace;
use liblumen_rt::error::ErlangException;
use liblumen_rt::function::{self, ErlangResult, ModuleFunctionArity};
use liblumen_rt::term::*;

use crate::scheduler;

#[export_name = "erlang:apply/2"]
pub extern "C-unwind" fn apply2(term: OpaqueTerm, arglist: OpaqueTerm) -> ErlangResult {
    let mut args = SmallVec::<[OpaqueTerm; 3]>::new();
    let callee = match (term.into(), arglist.into()) {
        (Term::Closure(fun), Term::Nil) => fun,
        (Term::Closure(fun), Term::Cons(ptr)) => {
            dbg!(&fun);
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

#[allow(improper_ctypes_definitions)]
#[export_name = "erlang:display/1"]
pub extern "C-unwind" fn display(term: OpaqueTerm) -> ErlangResult {
    let term: Term = term.into();
    println!("{}", &term);
    ErlangResult::Ok(true.into())
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
#[export_name = "erlang:=:=/2"]
pub extern "C-unwind" fn exact_eq(lhs: OpaqueTerm, rhs: OpaqueTerm) -> ErlangResult {
    let lhs: Term = lhs.into();
    let rhs: Term = rhs.into();
    ErlangResult::Ok(lhs.exact_eq(&rhs).into())
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
        Tuple::from_slice(&[tag.into(), reason.into()], proc)
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
    raise2(atoms::Badarg.into(), unsafe {
        NonNull::new_unchecked(Trace::into_raw(trace))
    })
}
