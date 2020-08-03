use std::mem;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use crate::distribution::external_term_format::try_split_at;
use crate::distribution::nodes::node;

use super::{arc_node, u16, u32, u64};

pub fn decode<'a>(
    process: &Process,
    safe: bool,
    bytes: &'a [u8],
) -> InternalResult<(Term, &'a [u8])> {
    let (u32_len_u16, after_len_bytes) = u16::decode(bytes)?;
    let len_usize = (u32_len_u16 as usize) * mem::size_of::<u32>();

    let (arc_node, after_node_bytes) = arc_node::decode(safe, after_len_bytes)?;
    // TODO use creation to differentiate respawned nodes
    let (_creation, after_creation_bytes) = u32::decode(after_node_bytes)?;

    try_split_at(after_creation_bytes, len_usize).and_then(|(id_bytes, after_id_bytes)| {
        if arc_node == node::arc_node() {
            let (scheduler_id_u32, after_scheduler_id_bytes) = u32::decode(id_bytes)?;
            let (number_u64, _) = u64::decode(after_scheduler_id_bytes)?;

            let reference =
                process.reference_from_scheduler(scheduler_id_u32.into(), number_u64)?;

            Ok((reference, after_id_bytes))
        } else {
            unimplemented!()
        }
    })
}
