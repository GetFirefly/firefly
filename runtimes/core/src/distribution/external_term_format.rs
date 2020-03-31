mod arc_node;
mod atom;
mod atom_utf8;
mod big;
mod binary;
mod bit_binary;
mod export;
mod f64;
mod i32;
mod integer;
mod isize;
mod list;
mod map;
mod new_float;
mod new_function;
mod new_pid;
mod newer_reference;
mod pid;
mod sign;
mod small_atom;
mod small_atom_utf8;
mod small_integer;
mod string;
pub mod term;
mod tuple;
mod u16;
mod u32;
mod u64;
mod u8;
pub mod version;

use std::backtrace::Backtrace;
use std::convert::TryInto;
use std::fmt::{self, Display};
use std::sync::Arc;

use anyhow::*;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use thiserror::Error;

use liblumen_alloc::erts::exception::{ArcError, InternalException, InternalResult};
use liblumen_alloc::erts::term::closure::Creator;
use liblumen_alloc::erts::term::prelude::{Pid as LocalPid, *};
use liblumen_alloc::erts::{Node, Process};
use liblumen_alloc::CloneToProcess;

use crate::distribution::nodes::node;

pub fn try_split_at(bytes: &[u8], mid: usize) -> InternalResult<(&[u8], &[u8])> {
    let available = bytes.len();

    if mid <= available {
        Ok(bytes.split_at(mid))
    } else {
        {
            Err(DecodeError::NotEnoughBytes {
                available,
                needed: mid,
                backtrace: Backtrace::capture(),
            }
            .into())
        }
    }
}

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("needed {needed} bytes, but only {available} available")]
    NotEnoughBytes {
        needed: usize,
        available: usize,
        backtrace: Backtrace,
    },
    #[error("unexpected version ({version})")]
    UnexpectedVersion { version: u8, backtrace: Backtrace },
    #[error("unexpected tag ({tag})")]
    UnexpectedTag { tag: Tag, backtrace: Backtrace },
}

impl From<DecodeError> for InternalException {
    fn from(decode_error: DecodeError) -> Self {
        ArcError::from_err(decode_error).into()
    }
}

#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum Tag {
    NewFloat = 70,
    BitBinary = 77,
    AtomCacheReference = 82,
    NewPID = 88,
    NewPort = 89,
    NewerReference = 90,
    SmallInteger = 97,
    Integer = 98,
    Float = 99,
    Function = 117,
    Atom = 100,
    Reference = 101,
    Port = 102,
    PID = 103,
    SmallTuple = 104,
    LargeTuple = 105,
    Nil = 106,
    String = 107,
    List = 108,
    Binary = 109,
    SmallBig = 110,
    LargeBig = 111,
    NewFunction = 112,
    Export = 113,
    NewReference = 114,
    SmallAtom = 115,
    Map = 116,
    AtomUTF8 = 118,
    SmallAtomUTF8 = 119,
}

impl Tag {
    pub fn decode(bytes: &[u8]) -> InternalResult<(Self, &[u8])> {
        let (tag_u8, after_tag_bytes) = u8::decode(bytes)?;
        let tag = tag_u8
            .try_into()
            .with_context(|| format!("tag ({})", tag_u8))?;

        Ok((tag, after_tag_bytes))
    }
}

impl Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub enum Pid {
    Local(LocalPid),
    External(ExternalPid),
}

impl Pid {
    fn decode(safe: bool, bytes: &[u8]) -> InternalResult<(Self, &[u8])> {
        let (tag, after_tag_bytes) = Tag::decode(bytes)?;

        match tag {
            Tag::PID => pid::decode_pid(safe, after_tag_bytes),
            Tag::NewPID => new_pid::decode_pid(safe, after_tag_bytes),
            _ => Err(DecodeError::UnexpectedTag {
                tag,
                backtrace: Backtrace::capture(),
            })
            .with_context(|| format!("Expected tags are {:?} or {:?}", Tag::PID, Tag::NewPID))
            .map_err(|error| error.into()),
        }
    }

    fn new(arc_node: Arc<Node>, id: u32, serial: u32) -> Result<Self, InvalidPidError> {
        let pid = if arc_node == node::arc_node() {
            let local_pid = LocalPid::new(id as usize, serial as usize)?;

            Pid::Local(local_pid)
        } else {
            let external_pid = ExternalPid::new(arc_node, id as usize, serial as usize)?;

            Pid::External(external_pid)
        };

        Ok(pid)
    }

    fn clone_to_process(&self, process: &Process) -> Term {
        match self {
            Pid::Local(local_pid) => local_pid.clone().into(),
            Pid::External(external_pid) => external_pid.clone_to_process(process),
        }
    }
}

impl Into<Creator> for Pid {
    fn into(self) -> Creator {
        match self {
            Pid::Local(local_pid) => Creator::Local(local_pid),
            Pid::External(external_pid) => Creator::External(external_pid),
        }
    }
}

// Private

fn decode_vec_term<'a>(
    process: &Process,
    safe: bool,
    bytes: &'a [u8],
    len: usize,
) -> InternalResult<(Vec<Term>, &'a [u8])> {
    let mut element_vec: Vec<Term> = Vec::with_capacity(len);
    let mut remaining_bytes = bytes;

    for _ in 0..len {
        let (element, after_element_bytes) = term::decode_tagged(process, safe, remaining_bytes)?;
        element_vec.push(element);
        remaining_bytes = after_element_bytes;
    }

    Ok((element_vec, remaining_bytes))
}
