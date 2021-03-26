#![allow(dead_code)]
//! This crate provides facilities for reading and extracting data from BEAM files
//!
//! # Examples
//!
//! Read a BEAM file:
//!
//!
//!     use liblumen_beam::beam::reader::StandardBeamFile;
//!     use liblumen_beam::beam::chunk::Chunk;
//!
//!     let beam = StandardBeamFile::from_file("tests/testdata/reader/test.beam").unwrap();
//!
//!     assert_eq!(vec![b"Atom", b"Code", b"StrT", b"ImpT", b"ExpT", b"FunT", b"LitT",
//!                     b"LocT", b"Attr", b"CInf", b"Abst", b"Line"],
//!                beam.chunks().iter().map(|c| c.id()).collect::<Vec<_>>());
//!
//!
//! Write a BEAM file:
//!
//!
//!     use liblumen_beam::beam::chunk::{Chunk, RawChunk};
//!     use liblumen_beam::beam::reader::RawBeamFile;
//!
//!     // NOTE: The following chunk is malformed
//!     let chunk = RawChunk{id: *b"Atom", data: Vec::new()};
//!     let mut beam = RawBeamFile::new();
//!     beam.push_chunk(chunk);
//!     beam.to_file("my.beam").unwrap();
pub mod chunk;
pub mod parts;

mod beam_file;

#[cfg(test)]
mod test;

pub use self::beam_file::BeamFile;

pub type RawBeamFile = BeamFile<chunk::RawChunk>;
pub type StandardBeamFile = BeamFile<chunk::StandardChunk>;
pub type Result<T> = std::result::Result<T, ReadError>;

#[derive(Debug)]
pub enum ReadError {
    FileError(std::io::Error),
    InvalidString(std::str::Utf8Error),
    UnexpectedMagicNumber([u8; 4]),
    UnexpectedFormType([u8; 4]),
    UnexpectedChunk { id: chunk::Id, expected: chunk::Id },
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::ReadError::*;
        match *self {
            FileError(ref x) => x.fmt(f),
            InvalidString(ref x) => x.fmt(f),
            UnexpectedMagicNumber(ref n) => write!(
                f,
                r#"Unexpected magic number {} (expected b"FOR1")"#,
                bytes_to_str(n)
            ),
            UnexpectedFormType(ref t) => write!(
                f,
                r#"Unexpected from type {} (expected b"BEAM")"#,
                bytes_to_str(t)
            ),
            UnexpectedChunk {
                ref id,
                ref expected,
            } => write!(
                f,
                "Unexpected chunk id {} (expected {})",
                bytes_to_str(id),
                bytes_to_str(expected)
            ),
        }
    }
}

impl std::error::Error for ReadError {
    fn cause(&self) -> Option<&dyn std::error::Error> {
        match *self {
            ReadError::FileError(ref x) => Some(x),
            ReadError::InvalidString(ref x) => Some(x),
            _ => None,
        }
    }
}

impl std::convert::From<std::io::Error> for ReadError {
    fn from(err: std::io::Error) -> Self {
        ReadError::FileError(err)
    }
}

impl std::convert::From<std::str::Utf8Error> for ReadError {
    fn from(err: std::str::Utf8Error) -> Self {
        ReadError::InvalidString(err)
    }
}

fn bytes_to_str(bytes: &[u8]) -> String {
    std::str::from_utf8(bytes)
        .map(|x| format!("b{:?}", x))
        .unwrap_or_else(|_| format!("{:?}", bytes))
}
