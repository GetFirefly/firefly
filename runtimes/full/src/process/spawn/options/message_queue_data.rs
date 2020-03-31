use std::convert::{TryFrom, TryInto};

use anyhow::Context;

use liblumen_alloc::erts::term::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum MessageQueueData {
    OnHeap,
    OffHeap,
}

impl Default for MessageQueueData {
    fn default() -> Self {
        MessageQueueData::OnHeap
    }
}

impl TryFrom<Term> for MessageQueueData {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let atom: Atom = term
            .try_into()
            .context("message_queue_data is not an atom")?;

        match atom.name() {
            "off_heap" => Ok(Self::OffHeap),
            "on_heap" => Ok(Self::OnHeap),
            name => Err(TryAtomFromTermError(name))
                .context("supported message_queue_data are off_heap or on_heap"),
        }
    }
}
