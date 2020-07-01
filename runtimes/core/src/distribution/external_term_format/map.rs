use hashbrown::HashMap;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::{term, u32};

pub fn decode<'a>(
    process: &Process,
    safe: bool,
    bytes: &'a [u8],
) -> InternalResult<(Term, &'a [u8])> {
    let (pair_len_u32, after_len_bytes) = u32::decode(bytes)?;
    let pair_len_usize = pair_len_u32 as usize;
    let mut hash_map: HashMap<Term, Term> = HashMap::with_capacity(pair_len_usize);
    let mut remaining_bytes = after_len_bytes;

    for _ in 0..pair_len_usize {
        let (key, after_key_bytes) = term::decode_tagged(process, safe, remaining_bytes)?;
        let (value, after_value_bytes) = term::decode_tagged(process, safe, after_key_bytes)?;
        hash_map.insert(key, value);
        remaining_bytes = after_value_bytes;
    }

    let map = process.map_from_hash_map(hash_map)?;

    Ok((map, remaining_bytes))
}
