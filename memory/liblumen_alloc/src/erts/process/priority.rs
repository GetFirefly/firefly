use core::convert::{TryFrom, TryInto};

use anyhow::Context;

use crate::erts::term::prelude::*;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Priority {
    Low,
    Normal,
    High,
    Max,
}

impl Default for Priority {
    fn default() -> Priority {
        Priority::Normal
    }
}

impl TryFrom<Term> for Priority {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let atom: Atom = term.try_into().context("priority is not an atom")?;

        match atom.name() {
            "low" => Ok(Priority::Low),
            "normal" => Ok(Priority::Normal),
            "high" => Ok(Priority::High),
            "max" => Ok(Priority::Max),
            name => Err(TryAtomFromTermError(name))
                .context("supported priorities are low, normal, high, or max"),
        }
    }
}
