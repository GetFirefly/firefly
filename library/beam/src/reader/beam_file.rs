use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::io::Read;
use std::io::Write;
use std::path::Path;

use super::chunk::{Chunk, ChunkId};
use super::ReadError;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

/// A BEAM File
///
/// ```
/// use firefly_beam::{Chunk, RawChunk, BeamFile};
///
/// let beam = BeamFile::<RawChunk>::from_file("tests/testdata/reader/test.beam").unwrap();
/// assert_eq!(
///     b"Atom",
///     beam.chunks().iter().nth(0).map(|c| c.id()).unwrap()
/// );
/// ```
#[derive(Debug)]
pub struct BeamFile<C> {
    chunks: HashMap<ChunkId, C>,
    order: Vec<ChunkId>,
}
impl<C: Chunk> BeamFile<C> {
    /// Creates a new empty BEAM file
    pub fn new() -> BeamFile<C> {
        let chunks: HashMap<ChunkId, C> = HashMap::new();
        let order: Vec<ChunkId> = Vec::new();
        BeamFile { chunks, order }
    }

    /// Adds a chunk to the BEAM file
    pub fn push_chunk(&mut self, chunk: C) {
        self.order.push(*chunk.id());
        self.chunks.insert(*chunk.id(), chunk);
    }

    /// Returns all chunks in the order they were encountered in the origin BEAM file
    ///
    /// ## Alternative Implementations
    /// - [`org.elixir_lang.beam.Beam#chunkCollection` in IntelliJ Elixir](https://github.com/KronicDeth/intellij-elixir/blob/
    ///   2f5c826040681e258e98c3e2f02b25985cd0766b/src/org/elixir_lang/beam/Beam.kt#L70) in Kotlin.
    pub fn chunks(&self) -> Vec<&C> {
        let mut result: Vec<&C> = Vec::new();
        for id in &self.order {
            match self.chunks.get(id) {
                None => continue,
                Some(c) => result.push(c),
            }
        }
        result
    }

    /// Returns a specific chunk from the BEAM file by id
    ///
    ///     use firefly_beam::{AtomChunk, StandardChunk, StandardBeamFile};
    ///
    ///     let beam = StandardBeamFile::from_file("tests/testdata/reader/test.beam").unwrap();
    ///     match beam.get_chunk(b"Atom") {
    ///         Some(StandardChunk::Atom(AtomChunk { atoms: _, is_unicode: false })) =>
    ///           {}
    ///         other =>
    ///           assert!(false, "assertion failed: got:{:?}", other)
    ///     }
    ///
    /// ## Alternative Implementations
    /// - [`org.elixir_lang.beam.Beam#chunk` in IntelliJ Elixir](https://github.com/KronicDeth/intellij-elixir/blob/
    ///   2f5c826040681e258e98c3e2f02b25985cd0766b/src/org/elixir_lang/beam/Beam.kt#L68-L69) in
    ///   Kotlin.
    pub fn get_chunk(&self, id: &ChunkId) -> Option<&C> {
        self.chunks.get(id)
    }

    /// Returns whichever chunk is the atom chunk, if it exists
    ///
    /// ## Alternative Implementations
    /// - [`org.elixir_lang.beam.Beam#atoms` in IntelliJ Elixir](https://github.com/KronicDeth/intellij-elixir/blob/
    ///   2f5c826040681e258e98c3e2f02b25985cd0766b/src/org/elixir_lang/beam/Beam.kt#L63-L65) in
    ///   Kotlin.
    pub fn atoms(&self) -> Option<&C> {
        match self.get_chunk(b"Atom") {
            Some(c) => Some(c),
            None => self.get_chunk(b"AtU8"),
        }
    }

    /// Strips a BEAM file of any chunks which are not required
    pub fn strip(&mut self) {
        self.chunks.retain(|_, ref mut c| c.is_required());
    }

    /// Strips a BEAM file using the provided predicate function.
    /// **NOTE:** You _must_ retain at least Code, ExpT, ImpT, StrT, and Line chunks
    pub fn strip_with<F>(&mut self, predicate: F)
    where
        F: Fn(&ChunkId, &C) -> bool,
    {
        self.chunks
            .retain(|&id, ref mut c| predicate(&id, &c) == false)
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ReadError> {
        let f = File::open(path)?;
        Self::from_reader(f)
    }

    pub fn from_reader<R: Read>(mut reader: R) -> Result<Self, ReadError> {
        let expected = Header::new(0);
        let header = Header::from_reader(&mut reader)?;
        if header.magic_number != expected.magic_number {
            return Err(ReadError::UnexpectedMagicNumber(header.magic_number));
        }
        if header.type_id != expected.type_id {
            return Err(ReadError::UnexpectedFormType(header.type_id));
        }

        let mut buf = vec![0; (header.payload_size - 4) as usize];
        reader.read_exact(&mut buf)?;

        let mut chunks: HashMap<ChunkId, C> = HashMap::new();
        let mut order: Vec<ChunkId> = Vec::new();
        let mut cursor = Cursor::new(&buf);
        while cursor.position() < buf.len() as u64 {
            let c = C::decode(&mut cursor)?;
            order.push(*c.id());
            chunks.insert(*c.id(), c);
        }
        Ok(BeamFile { chunks, order })
    }

    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let f = File::create(path)?;
        self.to_writer(f)
    }

    pub fn to_writer<W: Write>(&self, mut writer: W) -> anyhow::Result<()> {
        let mut buf = Vec::new();
        for chunk in self.chunks() {
            chunk.encode(&mut buf)?;
        }

        let header = Header::new(buf.len() as u32 + 4);
        header.to_writer(&mut writer)?;
        writer.write_all(&buf)?;
        Ok(())
    }
}

struct Header {
    magic_number: [u8; 4],
    payload_size: u32,
    type_id: [u8; 4],
}
impl Header {
    fn new(payload_size: u32) -> Self {
        Header {
            magic_number: *b"FOR1",
            payload_size,
            type_id: *b"BEAM",
        }
    }

    fn from_reader<R: Read>(mut reader: R) -> Result<Self, ReadError> {
        let mut header = Self::new(0);
        reader.read_exact(&mut header.magic_number)?;
        header.payload_size = reader.read_u32::<BigEndian>()?;
        reader.read_exact(&mut header.type_id)?;
        Ok(header)
    }

    fn to_writer<W: Write>(&self, mut writer: W) -> anyhow::Result<()> {
        writer.write_all(&self.magic_number)?;
        writer.write_u32::<BigEndian>(self.payload_size)?;
        writer.write_all(&self.type_id)?;
        Ok(())
    }
}
