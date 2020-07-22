use crate::erts::term::prelude::*;

use liblumen_core::sys::Endianness;

#[repr(C)]
pub struct BinaryMatchResult {
    // The value matched by the match operation
    pub value: Term,
    // The rest of the binary (typically a MatchContext)
    pub rest: Term,
    // Whether the match was successful or not
    pub success: bool,
}
impl BinaryMatchResult {
    pub fn success(value: Term, rest: Term) -> Self {
        Self {
            value,
            rest,
            success: true,
        }
    }

    pub fn failed() -> Self {
        Self {
            value: Term::NONE,
            rest: Term::NONE,
            success: false,
        }
    }
}

pub fn match_raw<B>(bin: B, unit: u8, size: Option<usize>) -> Result<BinaryMatchResult, ()>
where
    B: Bitstring + MaybePartialByte,
{
    let size = size.unwrap_or(0);
    let total_bits = bin.total_bit_len();

    if size == 0 && total_bits == 0 {
        // TODO: t = bin_to_term
        let t = Term::NONE;
        return Ok(BinaryMatchResult::success(Term::NONE, t));
    }

    unimplemented!()
}
