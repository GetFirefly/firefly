use super::Id;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

use crate::beam::reader::{ReadError, Result};

pub struct Header {
    pub chunk_id: Id,
    pub data_size: u32,
}
impl Header {
    pub fn new(chunk_id: &Id, data_size: u32) -> Self {
        Header {
            chunk_id: *chunk_id,
            data_size,
        }
    }
    /// Alternative Implementations
    /// - [In `org.elixir_lang.bean.chunk.Chunk.from` in IntelliJ
    ///   Elixir](https://github.com/KronicDeth/intellij-elixir/blob/
    ///   2f5c826040681e258e98c3e2f02b25985cd0766b/src/org/elixir_lang/beam/chunk/Chunk.java#
    ///   L36-L40) in Java.
    pub fn decode<R: std::io::Read>(mut reader: R) -> std::io::Result<Self> {
        let mut id = [0; 4];
        reader.read_exact(&mut id)?;
        let size = reader.read_u32::<BigEndian>()?;
        Ok(Header::new(&id, size))
    }
    pub fn encode<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        writer.write_all(&self.chunk_id)?;
        writer.write_u32::<BigEndian>(self.data_size)?;
        Ok(())
    }
}

/// ## Alternative Implementations
/// - [In `org.elixir_lang.bean.chunk.Chunk.from` in IntelliJ
///   Elixir](https://github.com/KronicDeth/intellij-elixir/blob/
///   2f5c826040681e258e98c3e2f02b25985cd0766b/src/org/elixir_lang/beam/chunk/Chunk.java#L45) in
///   Java.
pub fn padding_size(data_size: u32) -> u32 {
    (4 - data_size % 4) % 4
}

pub fn check_chunk_id(passed: &Id, expected: &Id) -> Result<()> {
    if passed != expected {
        Err(ReadError::UnexpectedChunk {
            id: *passed,
            expected: *expected,
        })
    } else {
        Ok(())
    }
}
