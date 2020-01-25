pub mod stack;

use alloc::sync::Arc;
use core::fmt::{self, Debug};

use std::cmp::*;
use std::hash::{Hash, Hasher};

use crate::erts::exception::{Exception, SystemException};
use crate::erts::process::Process;
use crate::location::Location;

pub type Result = core::result::Result<(), SystemException>;
pub type Code = fn(&Arc<Process>) -> Result;

#[derive(Clone, Copy)]
pub struct LocatedCode {
    pub code: Code,
    pub location: Location,
}

impl LocatedCode {
    fn code_address(&self) -> usize {
        self.code as usize
    }
}

impl Debug for LocatedCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:x} at {}", self.code_address(), self.location)
    }
}

impl Eq for LocatedCode {}

impl Hash for LocatedCode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.code_address().hash(state);
        self.location.hash(state);
    }
}

impl Ord for LocatedCode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.code_address()
            .cmp(&other.code_address())
            .then_with(|| self.location.cmp(&other.location))
    }
}

impl PartialEq for LocatedCode {
    fn eq(&self, other: &Self) -> bool {
        (self.code_address() == other.code_address()) && (self.location == other.location)
    }
}

impl PartialOrd for LocatedCode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn result_from_exception<P>(process: P, stack_used: usize, exception: Exception) -> Result
where
    P: AsRef<Process>,
{
    match exception {
        Exception::Runtime(err) => {
            let process_ref = process.as_ref();
            process_ref.stack_popn(stack_used);
            process_ref.exception(err);

            Ok(())
        }
        Exception::System(err) => Err(err),
    }
}
