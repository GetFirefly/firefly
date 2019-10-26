use core::convert::{From, TryInto};

use crate::erts::exception::runtime;

use super::integer::TryIntoIntegerError;
use super::prelude::*;
use super::arch::{arch_32, arch_64, arch_x86_64};

/// This error type is used to indicate a type conversion error
#[derive(Debug)]
pub struct TypeError;

/// This error type is used to indicate that a value is an invalid boolean
#[derive(Debug)]
pub enum BoolError {
    Type,
    NotABooleanName,
}

macro_rules! impl_term_conversions {
    ($raw:ty) => {
        impl From<SmallInteger> for $raw {
            #[inline]
            fn from(i: SmallInteger) -> Self {
                i.encode().unwrap()
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
            T: Encode<$raw>,
        {
            #[inline]
            default fn from(boxed: Boxed<T>) -> Self {
                boxed.encode().unwrap()
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
            type Error = runtime::Exception;

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

        impl TryInto<Boxed<Reference>> for $raw {
            type Error = TypeError;

            fn try_into(self) -> Result<Boxed<Reference>, Self::Error> {
                self.decode().unwrap().try_into()
            }
        }
    }
}

impl_term_conversions!(arch_32::RawTerm);
impl_term_conversions!(arch_64::RawTerm);
impl_term_conversions!(arch_x86_64::RawTerm);
