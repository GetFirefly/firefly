pub mod receive;

use std::convert::TryInto;
use std::panic;

use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use crate::process::current_process;
use crate::registry;

extern "Rust" {
    #[link_name = "__scheduler_stop_waiting"]
    fn stop_waiting(proc: &Process);
}

#[export_name = "__lumen_builtin_self"]
pub extern "C" fn builtin_self() -> Term {
    current_process().pid_term()
}

#[export_name = "__lumen_builtin_send"]
pub extern "C" fn builtin_send(to_term: Term, msg: Term) -> Term {
    let result = panic::catch_unwind(|| {
        let decoded_result: Result<Pid, _> = to_term.decode().unwrap().try_into();
        if let Ok(to) = decoded_result {
            let p = current_process();
            let self_pid = p.pid();
            if self_pid == to {
                p.send_from_self(msg);
                return msg;
            } else {
                if let Some(ref to_proc) = registry::pid_to_process(&to) {
                    if let Ok(resume) = to_proc.send_from_other(msg) {
                        if resume {
                            unsafe {
                                stop_waiting(to_proc);
                            }
                        }
                        return msg;
                    }
                }
            }
        }

        Term::NONE
    });
    if let Ok(res) = result {
        res
    } else {
        Term::NONE
    }
}

/// Strict equality
#[export_name = "__lumen_builtin_cmp.eq"]
pub extern "C" fn builtin_cmpeq(lhs: Term, rhs: Term) -> bool {
    let result = panic::catch_unwind(|| {
        if let Ok(left) = lhs.decode() {
            if let Ok(right) = rhs.decode() {
                left.exact_eq(&right)
            } else {
                false
            }
        } else {
            if lhs.is_none() && rhs.is_none() {
                true
            } else {
                false
            }
        }
    });
    if let Ok(res) = result {
        res
    } else {
        false
    }
}

#[export_name = "__lumen_builtin_cmp.neq"]
pub extern "C" fn builtin_cmpneq(lhs: Term, rhs: Term) -> bool {
    !builtin_cmpeq(lhs, rhs)
}

macro_rules! comparison_builtin {
    ($name:expr, $alias:ident, $op:tt) => {
        #[export_name = $name]
        pub extern "C" fn $alias(lhs: Term, rhs: Term) -> bool {
            let result = panic::catch_unwind(|| {
                if let Ok(left) = lhs.decode() {
                    if let Ok(right) = rhs.decode() {
                        return left $op right;
                    }
                }
                false
            });
            if let Ok(res) = result {
                res
            } else {
                false
            }
        }
    }
}

comparison_builtin!("__lumen_builtin_cmp.lt",  builtin_cmp_lt,  <);
comparison_builtin!("__lumen_builtin_cmp.lte", builtin_cmp_lte, <=);
comparison_builtin!("__lumen_builtin_cmp.gt",  builtin_cmp_gt,  >);
comparison_builtin!("__lumen_builtin_cmp.gte", builtin_cmp_gte, >=);

macro_rules! math_builtin {
    ($name:expr, $alias:ident, $trait:tt, $op:ident) => {
        #[export_name = $name]
        pub extern "C" fn $alias(lhs: Term, rhs: Term) -> Term {
            use std::ops::*;
            let result = panic::catch_unwind(|| {
                let l = lhs.decode().unwrap();
                let r = rhs.decode().unwrap();
                match (l, r) {
                    (TypedTerm::SmallInteger(li), TypedTerm::SmallInteger(ri)) => {
                        current_process().integer(li.$op(ri)).unwrap()
                    }
                    (TypedTerm::SmallInteger(li), TypedTerm::Float(ri)) => {
                        let li: f64 = li.into();
                        let f = <f64 as $trait<f64>>::$op(li, ri.value());
                        current_process().float(f).unwrap()
                    }
                    (TypedTerm::SmallInteger(li), TypedTerm::BigInteger(ri)) => {
                        let li: BigInteger = li.into();
                        current_process().integer(li.$op(ri.as_ref())).unwrap()
                    }
                    (TypedTerm::Float(li), TypedTerm::Float(ri)) => {
                        let f = <f64 as $trait<f64>>::$op(li.value(), ri.value());
                        current_process().float(f).unwrap()
                    }
                    (TypedTerm::Float(li), TypedTerm::SmallInteger(ri)) => {
                        let ri: f64 = ri.into();
                        let f = <f64 as $trait<f64>>::$op(li.value(), ri);
                        current_process().float(f).unwrap()
                    }
                    (TypedTerm::Float(li), TypedTerm::BigInteger(ri)) => {
                        let ri: f64 = ri.as_ref().into();
                        let f = <f64 as $trait<f64>>::$op(li.value(), ri);
                        current_process().float(f).unwrap()
                    }
                    (TypedTerm::BigInteger(li), TypedTerm::SmallInteger(ri)) => {
                        let ri: BigInteger = ri.into();
                        current_process().integer(li.as_ref().$op(ri)).unwrap()
                    }
                    (TypedTerm::BigInteger(li), TypedTerm::Float(ri)) => {
                        let li: f64 = li.as_ref().into();
                        let f = <f64 as $trait<f64>>::$op(li, ri.value());
                        current_process().float(f).unwrap()
                    }
                    (TypedTerm::BigInteger(li), TypedTerm::BigInteger(ri)) => {
                        current_process().integer(li.$op(ri)).unwrap()
                    }
                    _ => panic!("expected numeric argument to builtin '{}'", $name),
                }
            });
            if let Ok(res) = result {
                res
            } else {
                Term::NONE
            }
        }
    }
}

macro_rules! integer_math_builtin {
    ($name:expr, $alias:ident, $op:ident) => {
        #[export_name = $name]
        pub extern "C" fn $alias(lhs: Term, rhs: Term) -> Term {
            use std::ops::*;
            let result = panic::catch_unwind(|| {
                let l = lhs.decode().unwrap();
                let r = rhs.decode().unwrap();
                let li: Integer = l.try_into().unwrap();
                let ri: Integer = r.try_into().unwrap();
                let result = li.$op(ri);
                current_process().integer(result).unwrap()
            });
            if let Ok(res) = result {
                res
            } else {
                Term::NONE
            }
        }
    }
}

math_builtin!("__lumen_builtin_math.add", builtin_math_add, Add, add);
math_builtin!("__lumen_builtin_math.sub", builtin_math_sub, Sub, sub);
math_builtin!("__lumen_builtin_math.mul", builtin_math_mul, Mul, mul);
math_builtin!("__lumen_builtin_math.fdiv", builtin_math_fdiv, Div, div);

integer_math_builtin!("__lumen_builtin_math.div", builtin_math_div, div);
integer_math_builtin!("__lumen_builtin_math.rem", builtin_math_rem, rem);
integer_math_builtin!("__lumen_builtin_math.bsl", builtin_math_bsl, shl);
integer_math_builtin!("__lumen_builtin_math.bsr", builtin_math_bsr, shr);
integer_math_builtin!("__lumen_builtin_math.band", builtin_math_band, bitand);
integer_math_builtin!("__lumen_builtin_math.bor", builtin_math_bor, bitor);
integer_math_builtin!("__lumen_builtin_math.bxor", builtin_math_bxor, bitxor);

/// Capture the current stack trace
#[export_name = "__lumen_builtin_trace_capture"]
pub extern "C" fn builtin_trace_capture() -> Term {
    // TODO:
    Term::NIL
}
