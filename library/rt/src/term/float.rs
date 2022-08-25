use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::Deref;

use super::BigInteger;

#[derive(Copy, Clone, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Float(pub f64);
impl Float {
    const I64_UPPER_BOUNDARY: f64 = (1i64 << f64::MANTISSA_DIGITS) as f64;
    const I64_LOWER_BOUNDARY: f64 = (-1i64 << f64::MANTISSA_DIGITS) as f64;

    #[inline]
    pub fn as_f64(&self) -> f64 {
        self.0
    }
}
impl fmt::Debug for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}
impl fmt::Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
impl Into<f64> for Float {
    #[inline(always)]
    fn into(self) -> f64 {
        self.0
    }
}
impl From<f64> for Float {
    #[inline(always)]
    fn from(f: f64) -> Self {
        Self(f)
    }
}
impl TryInto<i64> for Float {
    type Error = ();

    fn try_into(self) -> Result<i64, Self::Error> {
        if self.is_nan() || self.is_infinite() {
            return Err(());
        }
        Ok(unsafe { self.0.to_int_unchecked() })
    }
}
impl Hash for Float {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state)
    }
}
impl Deref for Float {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl PartialEq<i64> for Float {
    fn eq(&self, y: &i64) -> bool {
        match self.0 {
            x if x.is_infinite() => false,
            x if x >= Self::I64_UPPER_BOUNDARY || x <= Self::I64_LOWER_BOUNDARY => {
                // We're out of the range where f64 is more precise than an i64,
                // so cast the float to integer and comapre.
                //
                // # Safety
                //
                // We've guarded against infinite values, the float cannot be NaN
                // due to our encoding scheme, and we know the value can be represented
                // in i64, so this is guaranteed safe.
                let x: i64 = unsafe { x.to_int_unchecked() };
                x.eq(y)
            }
            x => x.eq(&(*y as f64)),
        }
    }
}
impl PartialEq<BigInteger> for Float {
    fn eq(&self, y: &BigInteger) -> bool {
        let Some(y) = y.as_i64() else { return false; };
        self.eq(&y)
    }
}
impl PartialOrd<i64> for Float {
    fn partial_cmp(&self, y: &i64) -> Option<core::cmp::Ordering> {
        use core::cmp::Ordering;

        match self.0 {
            x if x.is_infinite() => {
                if x.is_sign_negative() {
                    Some(Ordering::Less)
                } else {
                    Some(Ordering::Greater)
                }
            }
            x if x >= Self::I64_UPPER_BOUNDARY || x <= Self::I64_LOWER_BOUNDARY => {
                // We're out of the range where f64 is more precise than an i64,
                // so cast the float to integer and comapre.
                //
                // # Safety
                //
                // We've guarded against infinite values, the float cannot be NaN
                // due to our encoding scheme, and we know the value can be represented
                // in i64, so this is guaranteed safe.
                let x: i64 = unsafe { x.to_int_unchecked() };
                Some(x.cmp(y))
            }
            x => x.partial_cmp(&(*y as f64)),
        }
    }
}
impl PartialOrd<BigInteger> for Float {
    fn partial_cmp(&self, y: &BigInteger) -> Option<core::cmp::Ordering> {
        use core::cmp::Ordering;

        match self.0 {
            x if x.is_infinite() => {
                if x.is_sign_negative() {
                    Some(Ordering::Less)
                } else {
                    Some(Ordering::Greater)
                }
            }
            _ => {
                let too_large = if y.is_negative() {
                    Ordering::Greater
                } else {
                    Ordering::Less
                };
                let Some(y) = y.as_i64() else { return Some(too_large); };
                self.partial_cmp(&y)
            }
        }
    }
}
