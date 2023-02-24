use std::io::Write;

use firefly_number::Int;
use firefly_rt::function::ErlangResult;
use firefly_rt::gc::{garbage_collect, Gc};
use firefly_rt::process::ProcessLock;
use firefly_rt::term::*;

use crate::{badarg, unwrap_or_badarg};

macro_rules! handle_arith_result {
    ($process:expr, $term:expr, $math:expr) => {
        match $math {
            Ok(Number::Float(n)) => ErlangResult::Ok(n.into()),
            Ok(Number::Integer(n)) => handle_safe_integer_arith_result!($process, n),
            Err(_) => badarg!($process, $term),
        }
    };
}

macro_rules! handle_integer_arith_result {
    ($process:expr, $term:expr, $math:expr) => {
        match $math {
            Ok(result) => handle_safe_integer_arith_result!($process, result),
            Err(_) => badarg!($process, $term),
        }
    };
}

macro_rules! handle_safe_integer_arith_result {
    ($process:expr, $math:expr) => {
        match $math {
            Int::Small(i) => ErlangResult::Ok(i.try_into().unwrap()),
            Int::Big(i) => {
                let p = $process;
                let i = BigInt::from(i);
                match Gc::new_uninit_in(p) {
                    Ok(mut empty) => unsafe {
                        empty.write(i);
                        ErlangResult::Ok(empty.assume_init().into())
                    },
                    Err(_) => {
                        assert!(garbage_collect(p, Default::default()).is_ok());
                        let boxed = Gc::new_in(i, p).unwrap();
                        ErlangResult::Ok(boxed.into())
                    }
                }
            }
        }
    };
}

#[export_name = "erlang:+/2"]
pub extern "C-unwind" fn plus2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    handle_arith_result!(process, lhs, l + r)
}

#[export_name = "erlang:-/1"]
pub extern "C-unwind" fn neg1(process: &mut ProcessLock, lhs: OpaqueTerm) -> ErlangResult {
    let l: Term = lhs.into();
    handle_arith_result!(process, lhs, -l)
}

#[export_name = "erlang:-/2"]
pub extern "C-unwind" fn minus2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    handle_arith_result!(process, lhs, l - r)
}

#[export_name = "erlang:*/2"]
pub extern "C-unwind" fn mul2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    handle_arith_result!(process, lhs, l * r)
}

#[export_name = "erlang://2"]
pub extern "C-unwind" fn divide2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    match l / r {
        Ok(result) => handle_arith_result!(process, lhs, result),
        Err(_) => badarg!(process, rhs),
    }
}

#[export_name = "erlang:div/2"]
pub extern "C-unwind" fn div2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let l: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let r: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_integer_arith_result!(process, rhs, l / r)
}

#[export_name = "erlang:rem/2"]
pub extern "C-unwind" fn rem2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let l: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let r: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_integer_arith_result!(process, rhs, l % r)
}

#[export_name = "erlang:bsl/2"]
pub extern "C-unwind" fn bsl2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let lhs: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let rhs: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_safe_integer_arith_result!(process, lhs << rhs)
}

#[export_name = "erlang:bsr/2"]
pub extern "C-unwind" fn bsr2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let lhs: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let rhs: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_safe_integer_arith_result!(process, lhs >> rhs)
}

#[export_name = "erlang:band/2"]
pub extern "C-unwind" fn band2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let lhs: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let rhs: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_safe_integer_arith_result!(process, lhs & rhs)
}

#[export_name = "erlang:bnot/1"]
pub extern "C-unwind" fn bnot1(process: &mut ProcessLock, lhs: OpaqueTerm) -> ErlangResult {
    let l: Term = lhs.into();
    match l {
        Term::Bool(b) => ErlangResult::Ok((!b).into()),
        Term::Int(i) => {
            let i = !i;
            let term: Result<OpaqueTerm, _> = i.try_into();
            match term {
                Ok(t) => ErlangResult::Ok(t),
                Err(_) => loop {
                    match Gc::new_uninit_in(process) {
                        Ok(mut empty) => unsafe {
                            empty.write(BigInt::from(i));
                            return ErlangResult::Ok(empty.assume_init().into());
                        },
                        Err(_) => {
                            assert!(garbage_collect(process, Default::default()).is_ok())
                        }
                    }
                },
            }
        }
        Term::BigInt(i) => {
            let i = !(i.as_ref());
            loop {
                match Gc::new_uninit_in(process) {
                    Ok(mut empty) => unsafe {
                        empty.write(i);
                        return ErlangResult::Ok(empty.assume_init().into());
                    },
                    Err(_) => {
                        assert!(garbage_collect(process, Default::default()).is_ok())
                    }
                }
            }
        }
        _ => badarg!(process, lhs),
    }
}

#[export_name = "erlang:bor/2"]
pub extern "C-unwind" fn bor2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let lhs: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let rhs: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_safe_integer_arith_result!(process, lhs | rhs)
}

#[export_name = "erlang:bxor/2"]
pub extern "C-unwind" fn bxor2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let lhs: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let rhs: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_safe_integer_arith_result!(process, lhs ^ rhs)
}

#[export_name = "erlang:abs/1"]
pub extern "C-unwind" fn abs(process: &mut ProcessLock, term: OpaqueTerm) -> ErlangResult {
    match term.into() {
        Term::Int(i) => {
            let i = i.abs();
            if i > Int::MAX_SMALL {
                let i = BigInt::from(i);
                loop {
                    match Gc::new_uninit_in(process) {
                        Ok(mut empty) => unsafe {
                            empty.write(i);
                            return ErlangResult::Ok(empty.assume_init().into());
                        },
                        Err(_) => {
                            assert!(garbage_collect(process, Default::default()).is_ok());
                        }
                    }
                }
            } else {
                ErlangResult::Ok(Term::Int(i).into())
            }
        }
        Term::BigInt(i) => {
            let i = i.abs();
            loop {
                match Gc::new_uninit_in(process) {
                    Ok(mut empty) => unsafe {
                        empty.write(i);
                        return ErlangResult::Ok(empty.assume_init().into());
                    },
                    Err(_) => {
                        assert!(garbage_collect(process, Default::default()).is_ok());
                    }
                }
            }
        }
        Term::Float(f) => ErlangResult::Ok(Term::Float(f.abs()).into()),
        _ => badarg!(process, term),
    }
}

#[export_name = "erlang:list_to_atom/1"]
pub extern "C-unwind" fn list_to_atom(process: &mut ProcessLock, term: OpaqueTerm) -> ErlangResult {
    match term.into() {
        Term::Nil => return ErlangResult::Ok(atoms::Empty.into()),
        Term::Cons(cons) => {
            if let Some(s) = cons.as_ref().to_string() {
                return ErlangResult::Ok(Atom::str_to_term(&s));
            }
        }
        _ => (),
    }
    badarg!(process, term)
}

#[export_name = "erlang:binary_to_list/1"]
pub extern "C-unwind" fn binary_to_list(
    process: &mut ProcessLock,
    term: OpaqueTerm,
) -> ErlangResult {
    let t: Term = term.into();
    if let Some(bits) = t.as_bitstring() {
        assert!(bits.is_binary());
        assert!(bits.is_aligned());

        let bytes = unsafe { bits.as_bytes_unchecked() };
        let result = match core::str::from_utf8(bytes).ok() {
            Some(s) => Cons::charlist_from_str(s, process).unwrap(),
            None => Cons::from_bytes(bytes, process).unwrap(),
        };
        match result {
            None => ErlangResult::Ok(Term::Nil.into()),
            Some(cons) => ErlangResult::Ok(cons.into()),
        }
    } else {
        badarg!(process, term)
    }
}

#[export_name = "erlang:display/1"]
pub extern "C-unwind" fn display(_process: &mut ProcessLock, term: OpaqueTerm) -> ErlangResult {
    let term: Term = term.into();
    println!("{}", &term);
    ErlangResult::Ok(true.into())
}

#[export_name = "erlang:debug/1"]
pub extern "C-unwind" fn debug(_process: &mut ProcessLock, term: OpaqueTerm) -> ErlangResult {
    let term: Term = term.into();
    println!("{:?}", &term);
    ErlangResult::Ok(true.into())
}

#[export_name = "erlang:display_nl/0"]
pub extern "C-unwind" fn display_nl(_process: &mut ProcessLock) -> ErlangResult {
    println!();
    ErlangResult::Ok(true.into())
}

#[export_name = "erlang:display_string/1"]
pub extern "C-unwind" fn display_string(
    process: &mut ProcessLock,
    term: OpaqueTerm,
) -> ErlangResult {
    let list: Term = term.into();
    match list {
        Term::Nil => ErlangResult::Ok(true.into()),
        Term::Cons(cons) => {
            match cons.as_ref().to_string() {
                Some(ref s) => print!("{}", s),
                None => badarg!(process, term),
            }
            ErlangResult::Ok(true.into())
        }
        _other => badarg!(process, term),
    }
}

#[export_name = "erlang:puts/1"]
pub extern "C-unwind" fn puts(_process: &mut ProcessLock, printable: OpaqueTerm) -> ErlangResult {
    let printable: Term = printable.into();

    let bits = printable.as_bitstring().unwrap();
    assert!(bits.is_aligned());
    assert!(bits.is_binary());
    let bytes = unsafe { bits.as_bytes_unchecked() };
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(bytes).unwrap();
    ErlangResult::Ok(true.into())
}

fn make_reason<R: Into<OpaqueTerm>>(process: &mut ProcessLock, tag: Atom, reason: R) -> OpaqueTerm {
    Tuple::from_slice(&[tag.into(), reason.into()], process)
        .unwrap()
        .into()
}
