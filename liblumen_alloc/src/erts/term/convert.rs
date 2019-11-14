use core::convert::{From, TryInto, Infallible};

use thiserror::Error;

use crate::erts::exception::Exception;

use super::integer::TryIntoIntegerError;
use super::prelude::*;
use super::arch::{arch_32, arch_64, arch_x86_64};

/// This error type is used to indicate a type conversion error
#[derive(Error, Debug)]
#[error("invalid term type")]
pub struct TypeError;
impl From<Infallible> for TypeError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

/// This error type is used to indicate that a value is an invalid boolean
#[derive(Error, Debug)]
pub enum BoolError {
    #[error("invalid boolean conversion: wrong type")]
    Type,
    #[error("invalid boolean conversion: bad atom, expected 'true' or 'false'")]
    NotABooleanName,
}
impl From<Infallible> for BoolError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

macro_rules! impl_term_conversions {
    ($raw:ty) => {
        impl From<AnyPid> for $raw {
            fn from(pid: AnyPid) -> Self {
                pid.encode().unwrap()
            }
        }

        impl Encode<$raw> for AnyPid {
            fn encode(&self) -> Result<$raw, crate::erts::exception::Exception> {
                match self {
                    AnyPid::Local(pid) => pid.encode(),
                    AnyPid::External(pid) => pid.encode(),
                }
            }
        }

        impl From<SmallInteger> for $raw {
            #[inline]
            fn from(i: SmallInteger) -> Self {
                i.encode().unwrap()
            }
        }

        impl From<Pid> for $raw {
            #[inline]
            fn from(pid: Pid) -> Self {
                pid.encode().unwrap()
            }
        }

        impl From<u8> for $raw {
            #[inline]
            fn from(i: u8) -> Self {
                i.encode().unwrap()
            }
        }

        impl From<bool> for $raw {
            #[inline]
            fn from(b: bool) -> Self {
                let atom: Atom = b.into();
                atom.encode().unwrap()
            }
        }

        impl<T> From<Boxed<T>> for $raw
        where
            T: ?Sized + Encode<$raw>,
        {
            #[inline]
            default fn from(boxed: Boxed<T>) -> Self {
                boxed.encode().unwrap()
            }
        }

        impl<T> From<Option<Boxed<T>>> for $raw
        where
            T: Encode<$raw>,
        {
            #[inline]
            default fn from(boxed: Option<Boxed<T>>) -> Self {
                match boxed {
                    None => <$raw>::NIL,
                    Some(ptr) => ptr.encode().unwrap()
                }
            }
        }

        impl TryInto<bool> for $raw {
            type Error = BoolError;

            fn try_into(self) -> Result<bool, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<char> for $raw {
            type Error = TryIntoIntegerError;

            fn try_into(self) -> Result<char, Self::Error> {
                let self_u32: u32 = self
                    .try_into()
                    .map_err(|_| TryIntoIntegerError::OutOfRange)?;

                match core::char::from_u32(self_u32) {
                    Some(c) => Ok(c),
                    None => Err(TryIntoIntegerError::OutOfRange),
                }
            }
        }

        impl TryInto<f64> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<f64, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Float> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Float, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<isize> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<isize, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<u8> for $raw {
            type Error = TryIntoIntegerError;

            fn try_into(self) -> Result<u8, Self::Error> {
                let u: u64 = self.try_into()?;

                u.try_into().map_err(|_| TryIntoIntegerError::OutOfRange)
            }
        }

        impl TryInto<u32> for $raw {
            type Error = TryIntoIntegerError;

            fn try_into(self) -> Result<u32, Self::Error> {
                let u: u64 = self.try_into()?;

                u.try_into().map_err(|_| TryIntoIntegerError::OutOfRange)
            }
        }

        impl TryInto<u64> for $raw {
            type Error = TryIntoIntegerError;

            fn try_into(self) -> Result<u64, Self::Error> {
                match self.decode().unwrap() {
                    TypedTerm::SmallInteger(small_integer) => small_integer
                        .try_into()
                        .map_err(|_| TryIntoIntegerError::OutOfRange),
                    TypedTerm::BigInteger(big_integer) => big_integer.try_into(),
                    _ => Err(TryIntoIntegerError::Type),
                }
            }
        }

        impl TryInto<SmallInteger> for $raw {
            type Error = TryIntoIntegerError;

            fn try_into(self) -> Result<SmallInteger, Self::Error> {
                self.decode().unwrap().try_into().map_err(|_| TryIntoIntegerError::Type)
            }
        }

        impl TryInto<usize> for $raw {
            type Error = TryIntoIntegerError;

            fn try_into(self) -> Result<usize, Self::Error> {
                let u: u64 = self.try_into()?;

                u.try_into().map_err(|_| TryIntoIntegerError::OutOfRange)
            }
        }

        impl TryInto<num_bigint::BigInt> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<num_bigint::BigInt, Self::Error> {
                let option_big_int = match self.decode().unwrap() {
                    TypedTerm::SmallInteger(small_integer) => Some(small_integer.into()),
                    TypedTerm::BigInteger(big_integer) => {
                        let big_int: num_bigint::BigInt = big_integer.clone().into();

                        Some(big_int.clone())
                    }
                    _ => None,
                };

                match option_big_int {
                    Some(big_int) => Ok(big_int),
                    None => Err(TypeError),
                }
            }
        }

        impl TryInto<Vec<u8>> for $raw {
            type Error = Exception;

            fn try_into(self) -> Result<Vec<u8>, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Atom> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Atom, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Pid> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Pid, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Port> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Port, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Boxed<Resource>> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Boxed<Resource>, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Resource> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Resource, Self::Error> {
                let boxed: Boxed<Resource> = self.try_into()?;
                Ok(boxed.into())
            }
        }

        impl TryInto<Boxed<Reference>> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Boxed<Reference>, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Boxed<Tuple>> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Boxed<Tuple>, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Boxed<Cons>> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Boxed<Cons>, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Boxed<Map>> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Boxed<Map>, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Boxed<Closure>> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Boxed<Closure>, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Boxed<BigInteger>> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Boxed<BigInteger>, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Boxed<SubBinary>> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Boxed<SubBinary>, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }

        impl TryInto<Boxed<HeapBin>> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Boxed<HeapBin>, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }
    }
}

impl_term_conversions!(arch_32::RawTerm);
impl_term_conversions!(arch_64::RawTerm);
impl_term_conversions!(arch_x86_64::RawTerm);
