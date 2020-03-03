//! [External Term Format](http://erlang.org/doc/apps/erts/erl_ext_dist.html)
use core::convert::TryFrom;

use thiserror::Error;

pub enum Tag {
    NewFloat = 70,
    BitBinary = 77,
    SmallInteger = 97,
    Integer = 98,
    Atom = 100,
    SmallTuple = 104,
    EmptyList = 106,
    ByteList = 107,
    List = 108,
    Binary = 109,
    SmallBigInteger = 110,
    SmallAtomUTF8 = 119,
}

impl TryFrom<u8> for Tag {
    type Error = anyhow::Error;

    fn try_from(tag_byte: u8) -> Result<Self, Self::Error> {
        use crate::term::external_format::Tag::*;

        match tag_byte {
            70 => Ok(NewFloat),
            77 => Ok(BitBinary),
            97 => Ok(SmallInteger),
            98 => Ok(Integer),
            100 => Ok(Atom),
            104 => Ok(SmallTuple),
            106 => Ok(EmptyList),
            107 => Ok(ByteList),
            108 => Ok(List),
            109 => Ok(Binary),
            110 => Ok(SmallBigInteger),
            119 => Ok(SmallAtomUTF8),
            byte => Err(TryFromByteError(byte).into()),
        }
    }
}

#[derive(Debug, Error)]
#[error("byte ({0}) is not a tag")]
pub struct TryFromByteError(pub u8);
