//! This crate provides facilities for reading and extracting data from BEAM files
//!
//! # Examples
//!
//! Read a BEAM file:
//!
//!
//!     use firefly_beam::{StandardBeamFile, Chunk};
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
//!     use firefly_beam::{RawBeamFile, Chunk, RawChunk};
//!
//!     // NOTE: The following chunk is malformed
//!     let chunk = RawChunk{id: *b"Atom", data: Vec::new()};
//!     let mut beam = RawBeamFile::new();
//!     beam.push_chunk(chunk);
//!     beam.to_file("my.beam").unwrap();
mod beam_file;
mod chunk;
mod parts;
#[cfg(test)]
mod test;

pub use self::beam_file::BeamFile;
pub use self::chunk::*;
pub use self::parts::*;

pub type RawBeamFile = BeamFile<chunk::RawChunk>;
pub type StandardBeamFile = BeamFile<chunk::StandardChunk>;

#[derive(thiserror::Error, Debug)]
pub enum ReadError {
    #[error("error occurred while reading beam file: {0}")]
    FileError(#[from] std::io::Error),
    #[error("invalid utf8 string")]
    InvalidString(#[from] std::str::Utf8Error),
    #[error("unexpected magic number {}, expected b\"FOR1\"", bytes_to_str(.0))]
    UnexpectedMagicNumber([u8; 4]),
    #[error("unexpected from type {}, expected b\"BEAM\"", bytes_to_str(.0))]
    UnexpectedFormType([u8; 4]),
    #[error("unexpected chunk id {}, expected {}", bytes_to_str(.id), bytes_to_str(.expected))]
    UnexpectedChunk { id: ChunkId, expected: ChunkId },
    #[error("error occurred while reading chunk: {0}")]
    ChunkError(#[from] anyhow::Error),
}

fn bytes_to_str(bytes: &[u8]) -> String {
    std::str::from_utf8(bytes)
        .map(|x| format!("b{:?}", x))
        .unwrap_or_else(|_| format!("{:?}", bytes))
}
