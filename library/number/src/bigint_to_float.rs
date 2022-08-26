use num_bigint::{BigInt, Sign};

/// https://github.com/erlang/otp/blob/0de9ecd561bdc964f1c6436d240729b3952cdf3a/erts/emulator/beam/big.c#L1642-L1667
pub fn bigint_to_double(big: &BigInt) -> f64 {
    let (sign, data) = big.to_bytes_be();

    let mut d: f64 = 0.0;
    let dbase = ((!0u8) as f64) + 1.0;

    for digit in data.iter() {
        d = d * dbase + (*digit as f64);
    }

    if sign == Sign::Minus {
        -d
    } else {
        d
    }
}

#[cfg(test)]
mod tests {
    use super::bigint_to_double;
    use num_bigint::BigInt;

    #[test]
    fn test_bigint_to_double() {
        let bi = BigInt::from(100);
        assert!(bigint_to_double(&bi) == 100.0);

        let bi = BigInt::from(10000000000i64);
        assert!(bigint_to_double(&bi) == 10000000000.0);

        let bi = BigInt::from(1000000000000000000i64);
        assert!(bigint_to_double(&bi) == 1000000000000000000.0);
    }
}
