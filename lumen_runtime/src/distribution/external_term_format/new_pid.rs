use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::{arc_node, u32, Pid};

pub fn decode_pid<'a>(safe: bool, bytes: &'a [u8]) -> InternalResult<(Pid, &'a [u8])> {
    let (arc_node, after_node_bytes) = arc_node::decode(safe, bytes)?;
    let (id, after_id_bytes) = u32::decode(after_node_bytes)?;
    let (serial, after_serial_bytes) = u32::decode(after_id_bytes)?;
    // TODO use creation to differentiate respawned nodes
    let (_creation, after_creation_bytes) = u32::decode(after_serial_bytes)?;

    let pid = Pid::new(arc_node, id, serial)?;

    Ok((pid, after_creation_bytes))
}

pub fn decode_term<'a>(
    process: &Process,
    safe: bool,
    bytes: &'a [u8],
) -> InternalResult<(Term, &'a [u8])> {
    decode_pid(safe, bytes)
        .map(|(pid, after_pid_bytes)| (pid.clone_to_process(process), after_pid_bytes))
}
