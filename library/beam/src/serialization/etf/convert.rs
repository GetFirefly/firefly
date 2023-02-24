use firefly_number::{Float, Int, ToPrimitive};

use super::*;

pub trait TryAsRef<T> {
    fn try_as_ref(&self) -> Option<&T>;
}

impl<T> TryAsRef<T> for T {
    fn try_as_ref(&self) -> Option<&T> {
        Some(self)
    }
}

macro_rules! impl_term_try_as_ref {
    ($to:ident) => {
        impl_term_try_as_ref!($to, $to);
    };

    ($to:ident, $from:ident) => {
        impl TryAsRef<$to> for Term {
            fn try_as_ref(&self) -> Option<&$to> {
                match *self {
                    Term::$from(ref x) => Some(x),
                    _ => None,
                }
            }
        }
    };
}
impl_term_try_as_ref!(Atom);
impl_term_try_as_ref!(Int, Integer);
impl_term_try_as_ref!(Float);
impl_term_try_as_ref!(Pid);
impl_term_try_as_ref!(Port);
impl_term_try_as_ref!(Reference);
impl_term_try_as_ref!(ExternalFun);
impl_term_try_as_ref!(InternalFun);
impl_term_try_as_ref!(Binary);
impl_term_try_as_ref!(BitBinary);
impl_term_try_as_ref!(List);
impl_term_try_as_ref!(ImproperList);
impl_term_try_as_ref!(Tuple);
impl_term_try_as_ref!(Map);

macro_rules! impl_term_try_into {
    ($to:ident) => {
        impl_term_try_into!($to, $to);
    };

    ($to:ident, $from:ident) => {
        impl TryInto<$to> for Term {
            type Error = Self;

            fn try_into(self) -> Result<$to, Self> {
                match self {
                    Term::$from(x) => Ok(x),
                    _ => Err(self),
                }
            }
        }
    };
}
impl_term_try_into!(Atom);
impl_term_try_into!(Int, Integer);
impl_term_try_into!(Float);
impl_term_try_into!(Pid);
impl_term_try_into!(Port);
impl_term_try_into!(Reference);
impl_term_try_into!(ExternalFun);
impl_term_try_into!(InternalFun);
impl_term_try_into!(Binary);
impl_term_try_into!(BitBinary);
impl_term_try_into!(List);
impl_term_try_into!(ImproperList);
impl_term_try_into!(Tuple);
impl_term_try_into!(Map);

impl ToPrimitive for Term {
    fn to_i64(&self) -> Option<i64> {
        match *self {
            Term::Integer(ref x) => x.to_i64(),
            _ => None,
        }
    }
    fn to_u64(&self) -> Option<u64> {
        match *self {
            Term::Integer(ref x) => x.to_u64(),
            _ => None,
        }
    }
    fn to_f64(&self) -> Option<f64> {
        match *self {
            Term::Integer(ref x) => x.to_f64(),
            Term::Float(ref x) => x.to_f64(),
            _ => None,
        }
    }
}

impl firefly_number::ToBigInt for Term {
    fn to_bigint(&self) -> Option<firefly_number::BigInt> {
        match self {
            Term::Integer(ref x) => x.to_bigint(),
            _ => None,
        }
    }
}

impl firefly_number::ToBigUint for Term {
    fn to_biguint(&self) -> Option<firefly_number::BigUint> {
        match self {
            Term::Integer(ref x) => x.to_biguint(),
            _ => None,
        }
    }
}
