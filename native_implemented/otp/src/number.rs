use num_bigint::BigInt;

pub enum Operands {
    Bad,
    ISizes(isize, isize),
    Floats(f64, f64),
    BigInts(BigInt, BigInt),
}
