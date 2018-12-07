//! The `Chunk` trait and implementations of commonly used chunks.
//!
//! # References
//!
//! - [BEAM File Format](http://beam-wisdoms.clau.se/en/latest/indepth-beam-file.html)
//! - [`beam_lib`](http://erlang.org/doc/man/beam_lib.html)
//!
mod aux;

use std::io::Cursor;
use std::io::Read;
use std::io::Write;
use std::str;

use byteorder::BigEndian;
use byteorder::ReadBytesExt;
use byteorder::WriteBytesExt;
use libflate::zlib;

use super::parts;
use super::Result;

/// The identifier which indicates the type of a chunk.
pub type Id = [u8; 4];

/// The `Chunk` trait represents a type of chunk in a BEAM file.
pub trait Chunk {
    /// Returns the identifier of the chunk.
    fn id(&self) -> &Id;

    /// Returns whether or not this chunk is required in a BEAM file
    fn is_required(&self) -> bool {
        match self.id() {
            b"Atom" => true,
            b"AtU8" => true,
            b"Code" => true,
            b"StrT" => true,
            b"ImpT" => true,
            b"ExpT" => true,
            b"LitT" => false,
            b"LocT" => false,
            b"FunT" => false,
            b"Attr" => false,
            b"CInf" => false,
            b"Abst" => false,
            b"Dbgi" => false,
            b"Docs" => false,
            b"Line" => true,
            _ => false,
        }
    }

    /// Reads a chunk from `reader`.
    fn decode<R: Read>(mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        let header = aux::Header::decode(&mut reader)?;
        let mut buf = vec![0; header.data_size as usize];
        reader.read_exact(&mut buf)?;
        for _ in 0..aux::padding_size(header.data_size) {
            reader.read_u8()?;
        }

        Self::decode_data(&header.chunk_id, Cursor::new(&buf))
    }

    /// Reads a chunk which has the identifier `id` from `reader`.
    ///
    /// NOTICE: `reader` has no chunk header (i.e., the identifier and data size of the chunk).
    fn decode_data<R: Read>(id: &Id, reader: R) -> Result<Self>
    where
        Self: Sized;

    /// Writes the chunk to `writer`.
    fn encode<W: Write>(&self, mut writer: W) -> Result<()> {
        let mut buf = Vec::new();
        self.encode_data(&mut buf)?;
        aux::Header::new(self.id(), buf.len() as u32).encode(&mut writer)?;
        writer.write_all(&buf)?;
        for _ in 0..aux::padding_size(buf.len() as u32) {
            writer.write_u8(0)?;
        }
        Ok(())
    }

    /// Writes the data of the chunk to `writer`.
    ///
    /// NOTICE: The header (i.e., identifier and data size) of
    /// the chunk must not write in the function.
    fn encode_data<W: Write>(&self, writer: W) -> Result<()>;
}

/// A raw representation of a chunk.
///
/// This implementation does not interpret the data of a chunk
/// at the time of reading it from a BEAM file.
#[derive(Debug, PartialEq, Eq)]
pub struct RawChunk {
    /// The identifier of the chunk.
    pub id: Id,

    /// The bare data of the chunk.
    pub data: Vec<u8>,
}
impl Chunk for RawChunk {
    fn id(&self) -> &Id {
        &self.id
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        Ok(RawChunk { id: *id, data: buf })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_all(&self.data)?;
        Ok(())
    }
}

/// A representation of the `"Atom"` and `"AtU8"` chunks.
#[derive(Debug, PartialEq, Eq)]
pub struct AtomChunk {
    /// Whether or not this Atom chunk contains UTF-8 atoms
    pub is_unicode: bool,
    /// The list of atoms contained in a BEAM file.
    pub atoms: Vec<parts::Atom>,
}
impl Chunk for AtomChunk {
    fn id(&self) -> &Id {
        if self.is_unicode {
            b"AtU8"
        } else {
            b"Atom"
        }
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        // This chunk can be either Atom or AtU8
        let unicode;
        match aux::check_chunk_id(id, b"Atom") {
            Err(_) => {
                aux::check_chunk_id(id, b"AtU8")?;
                unicode = true;
            }
            Ok(_) => unicode = false,
        }
        let count = reader.read_u32::<BigEndian>()? as usize;
        let mut atoms = Vec::with_capacity(count);
        for _ in 0..count {
            let len = reader.read_u8()? as usize;
            let mut buf = vec![0; len];
            reader.read_exact(&mut buf)?;

            let name = str::from_utf8(&buf).map(|s| s.to_string())?;
            atoms.push(parts::Atom {
                name: name.to_string(),
            });
        }
        Ok(AtomChunk {
            is_unicode: unicode,
            atoms: atoms,
        })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_u32::<BigEndian>(self.atoms.len() as u32)?;
        for atom in &self.atoms {
            assert!(atom.name.len() < 0x100);
            writer.write_u8(atom.name.len() as u8)?;
            writer.write_all(atom.name.as_bytes())?;
        }
        Ok(())
    }
}

/// A representation of the `"Code"` chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct CodeChunk {
    /// Length of the information fields before code.
    pub info_size: u32,

    /// Instruction set version.
    pub version: u32,

    /// The highest opcode used in the code section.
    pub opcode_max: u32,

    /// The number of labels.
    pub label_count: u32,

    /// The number of functions.
    pub function_count: u32,

    /// The byte code.
    pub bytecode: Vec<u8>,
}
impl Chunk for CodeChunk {
    fn id(&self) -> &Id {
        b"Code"
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        aux::check_chunk_id(id, b"Code")?;
        let mut code = CodeChunk {
            info_size: reader.read_u32::<BigEndian>()?,
            version: reader.read_u32::<BigEndian>()?,
            opcode_max: reader.read_u32::<BigEndian>()?,
            label_count: reader.read_u32::<BigEndian>()?,
            function_count: reader.read_u32::<BigEndian>()?,
            bytecode: Vec::new(),
        };
        reader.read_to_end(&mut code.bytecode)?;
        Ok(code)
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_u32::<BigEndian>(self.info_size)?;
        writer.write_u32::<BigEndian>(self.version)?;
        writer.write_u32::<BigEndian>(self.opcode_max)?;
        writer.write_u32::<BigEndian>(self.label_count)?;
        writer.write_u32::<BigEndian>(self.function_count)?;
        writer.write_all(&self.bytecode)?;
        Ok(())
    }
}

/// A representation of the `"StrT"` chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct StrTChunk {
    /// Concatenated strings.
    pub strings: Vec<u8>,
}
impl Chunk for StrTChunk {
    fn id(&self) -> &Id {
        b"StrT"
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        aux::check_chunk_id(id, b"StrT")?;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        Ok(StrTChunk { strings: buf })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_all(&self.strings)?;
        Ok(())
    }
}

/// A representation of the `"ImpT"` chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct ImpTChunk {
    /// The list of imported functions.
    pub imports: Vec<parts::Import>,
}
impl Chunk for ImpTChunk {
    fn id(&self) -> &Id {
        b"ImpT"
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        aux::check_chunk_id(id, b"ImpT")?;
        let count = reader.read_u32::<BigEndian>()? as usize;
        let mut imports = Vec::with_capacity(count);
        for _ in 0..count {
            imports.push(parts::Import {
                module: reader.read_u32::<BigEndian>()?,
                function: reader.read_u32::<BigEndian>()?,
                arity: reader.read_u32::<BigEndian>()?,
            });
        }
        Ok(ImpTChunk { imports: imports })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_u32::<BigEndian>(self.imports.len() as u32)?;
        for import in &self.imports {
            writer.write_u32::<BigEndian>(import.module)?;
            writer.write_u32::<BigEndian>(import.function)?;
            writer.write_u32::<BigEndian>(import.arity)?;
        }
        Ok(())
    }
}

/// A representation of the `"ExpT"` chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct ExpTChunk {
    /// The list of exported functions.
    pub exports: Vec<parts::Export>,
}
impl Chunk for ExpTChunk {
    fn id(&self) -> &Id {
        b"ExpT"
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        aux::check_chunk_id(id, b"ExpT")?;
        let count = reader.read_u32::<BigEndian>()? as usize;
        let mut exports = Vec::with_capacity(count);
        for _ in 0..count {
            exports.push(parts::Export {
                function: reader.read_u32::<BigEndian>()?,
                arity: reader.read_u32::<BigEndian>()?,
                label: reader.read_u32::<BigEndian>()?,
            });
        }
        Ok(ExpTChunk { exports: exports })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_u32::<BigEndian>(self.exports.len() as u32)?;
        for export in &self.exports {
            writer.write_u32::<BigEndian>(export.function)?;
            writer.write_u32::<BigEndian>(export.arity)?;
            writer.write_u32::<BigEndian>(export.label)?;
        }
        Ok(())
    }
}

/// A representation of the `"LitT"` chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct LitTChunk {
    /// The list of literal terms.
    ///
    /// Each term is encoded in the [External Term Format]
    /// (http://erlang.org/doc/apps/erts/erl_ext_dist.html).
    pub literals: Vec<parts::ExternalTermFormatBinary>,
}
impl Chunk for LitTChunk {
    fn id(&self) -> &Id {
        b"LitT"
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        aux::check_chunk_id(id, b"LitT")?;
        let _uncompressed_size = reader.read_u32::<BigEndian>()?;
        let mut decoder = zlib::Decoder::new(reader)?;

        let count = decoder.read_u32::<BigEndian>()? as usize;
        let mut literals = Vec::with_capacity(count);
        for _ in 0..count {
            let literal_size = decoder.read_u32::<BigEndian>()? as usize;
            let mut buf = vec![0; literal_size];
            decoder.read_exact(&mut buf)?;
            literals.push(buf);
        }
        Ok(LitTChunk { literals: literals })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        let uncompressed_size = self
            .literals
            .iter()
            .fold(4, |acc, l| acc + 4 + l.len() as u32);
        writer.write_u32::<BigEndian>(uncompressed_size)?;

        let mut encoder = zlib::Encoder::new(writer)?;
        encoder.write_u32::<BigEndian>(self.literals.len() as u32)?;
        for literal in &self.literals {
            encoder.write_u32::<BigEndian>(literal.len() as u32)?;
            encoder.write_all(literal)?;
        }
        encoder.finish().into_result()?;
        Ok(())
    }
}

/// A representation of the `"LocT"` chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct LocTChunk {
    /// The list of local functions.
    pub locals: Vec<parts::Local>,
}
impl Chunk for LocTChunk {
    fn id(&self) -> &Id {
        b"LocT"
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        aux::check_chunk_id(id, b"LocT")?;
        let count = reader.read_u32::<BigEndian>()? as usize;
        let mut locals = Vec::with_capacity(count);
        for _ in 0..count {
            locals.push(parts::Local {
                function: reader.read_u32::<BigEndian>()?,
                arity: reader.read_u32::<BigEndian>()?,
                label: reader.read_u32::<BigEndian>()?,
            });
        }
        Ok(LocTChunk { locals: locals })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_u32::<BigEndian>(self.locals.len() as u32)?;
        for local in &self.locals {
            writer.write_u32::<BigEndian>(local.function)?;
            writer.write_u32::<BigEndian>(local.arity)?;
            writer.write_u32::<BigEndian>(local.label)?;
        }
        Ok(())
    }
}

/// A representation of the `"FunT"` chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct FunTChunk {
    /// The list of anonymous functions.
    pub functions: Vec<parts::Function>,
}
impl Chunk for FunTChunk {
    fn id(&self) -> &Id {
        b"FunT"
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        aux::check_chunk_id(id, b"FunT")?;
        let count = reader.read_u32::<BigEndian>()? as usize;
        let mut functions = Vec::with_capacity(count);
        for _ in 0..count {
            functions.push(parts::Function {
                function: reader.read_u32::<BigEndian>()?,
                arity: reader.read_u32::<BigEndian>()?,
                label: reader.read_u32::<BigEndian>()?,
                index: reader.read_u32::<BigEndian>()?,
                num_free: reader.read_u32::<BigEndian>()?,
                old_uniq: reader.read_u32::<BigEndian>()?,
            });
        }
        Ok(FunTChunk {
            functions: functions,
        })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_u32::<BigEndian>(self.functions.len() as u32)?;
        for f in &self.functions {
            writer.write_u32::<BigEndian>(f.function)?;
            writer.write_u32::<BigEndian>(f.arity)?;
            writer.write_u32::<BigEndian>(f.label)?;
            writer.write_u32::<BigEndian>(f.index)?;
            writer.write_u32::<BigEndian>(f.num_free)?;
            writer.write_u32::<BigEndian>(f.old_uniq)?;
        }
        Ok(())
    }
}

/// A representation of the `"Attr"` chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct AttrChunk {
    /// The attributes of a module (i.e., BEAM file).
    ///
    /// The value is equivalent to the result of the following erlang code.
    /// ```erlang
    /// term_to_binary(Module:module_info(attributes)).
    /// ```
    pub term: parts::ExternalTermFormatBinary,
}
impl Chunk for AttrChunk {
    fn id(&self) -> &Id {
        b"Attr"
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        aux::check_chunk_id(id, b"Attr")?;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        Ok(AttrChunk { term: buf })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_all(&self.term)?;
        Ok(())
    }
}

/// A representation of the `"CInf"` chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct CInfChunk {
    /// The compile information of a module (i.e., BEAM file).
    ///
    /// The value is equivalent to the result of the following erlang code.
    /// ```erlang
    /// term_to_binary(Module:module_info(compile)).
    /// ```
    pub term: parts::ExternalTermFormatBinary,
}
impl Chunk for CInfChunk {
    fn id(&self) -> &Id {
        b"CInf"
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        aux::check_chunk_id(id, b"CInf")?;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        Ok(CInfChunk { term: buf })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_all(&self.term)?;
        Ok(())
    }
}

/// A representation of the `"Abst"` chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct AbstChunk {
    /// The abstract code of a module (i.e., BEAM file).
    ///
    /// The value is encoded in the [External Term Format]
    /// (http://erlang.org/doc/apps/erts/erl_ext_dist.html) and
    /// represents [The Abstract Format](http://erlang.org/doc/apps/erts/absform.html).
    pub term: parts::ExternalTermFormatBinary,
}
impl Chunk for AbstChunk {
    fn id(&self) -> &Id {
        b"Abst"
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        aux::check_chunk_id(id, b"Abst")?;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        Ok(AbstChunk { term: buf })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_all(&self.term)?;
        Ok(())
    }
}

/// A representation of the `"Dbgi"` chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct DbgiChunk {
    /// The debug information for a module (i.e., BEAM file).
    ///
    /// Supercedes 'Abst' in recent versions of OTP by supporting arbitrary abstract formats.
    ///
    /// The value is encoded in the [External Term Format]
    /// (http://erlang.org/doc/apps/erts/erl_ext_dist.html) and
    /// represents custom debug information in the following term format:
    ///
    /// ```erlang
    /// {debug_info, {Backend, Data}}
    /// ```
    ///
    /// Where `Backend` is a module which implements `debug_info/4`, and is responsible for
    /// converting `Data` to different representations as described [here](http://erlang.org/doc/man/beam_lib.html#type-debug_info).
    /// Debug information can be used to reconstruct original source code.
    pub term: parts::ExternalTermFormatBinary,
}
impl Chunk for DbgiChunk {
    fn id(&self) -> &Id {
        b"Dbgi"
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        aux::check_chunk_id(id, b"Dbgi")?;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        Ok(DbgiChunk { term: buf })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_all(&self.term)?;
        Ok(())
    }
}

/// A representation of the `"Docs"` chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct DocsChunk {
    /// The 'Docs' chunk contains embedded module documentation, such as moduledoc/doc in Elixir
    ///
    /// The value is encoded in the [External Term Format]
    /// (http://erlang.org/doc/apps/erts/erl_ext_dist.html) and
    /// represents a term in the following format:
    ///
    /// ```erlang
    /// {Module, [{"Docs", DocsBin}]}
    /// ```
    ///
    /// Where `Module` is the documented module, and `DocsBin` is a binary in External Term Format
    /// containing the documentation. Currently, that decodes to:
    ///
    /// ```erlang
    /// {docs_v1, Anno, BeamLang, Format, ModuleDoc, Metadata, Docs}
    ///   where Anno :: erl_anno:anno(),
    ///         BeamLang :: erlang | elixir | lfe | alpaca | atom(),
    ///         Format :: binary(),
    ///         ModuleDoc :: doc_content(),
    ///         Metadata :: map(),
    ///         Docs :: [doc_element()],
    ///         signature :: [binary],
    ///         doc_content :: map(binary(), binary()) | none | hidden,
    ///         doc_element :: {{kind :: atom(), function :: atom(), arity}, Anno, signature, doc_content(), Metadata}
    /// ```
    ///
    pub term: parts::ExternalTermFormatBinary,
}
impl Chunk for DocsChunk {
    fn id(&self) -> &Id {
        b"Docs"
    }
    fn decode_data<R: Read>(id: &Id, mut reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        aux::check_chunk_id(id, b"Docs")?;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        Ok(DocsChunk { term: buf })
    }
    fn encode_data<W: Write>(&self, mut writer: W) -> Result<()> {
        writer.write_all(&self.term)?;
        Ok(())
    }
}

/// A representation of commonly used chunk.
///
/// ```
/// use beam_file::BeamFile;
/// use beam_file::chunk::{Chunk, StandardChunk};
///
/// let beam = BeamFile::<StandardChunk>::from_file("tests/testdata/test.beam").unwrap();
/// assert_eq!(b"Atom", beam.chunks().iter().nth(0).map(|c| c.id()).unwrap());
/// ```
#[derive(Debug, PartialEq, Eq)]
pub enum StandardChunk {
    Atom(AtomChunk),
    Code(CodeChunk),
    StrT(StrTChunk),
    ImpT(ImpTChunk),
    ExpT(ExpTChunk),
    LitT(LitTChunk),
    LocT(LocTChunk),
    FunT(FunTChunk),
    Attr(AttrChunk),
    CInf(CInfChunk),
    Abst(AbstChunk),
    Dbgi(DbgiChunk),
    Docs(DocsChunk),
    Unknown(RawChunk),
}
impl Chunk for StandardChunk {
    fn id(&self) -> &Id {
        use self::StandardChunk::*;
        match *self {
            Atom(ref c) => c.id(),
            Code(ref c) => c.id(),
            StrT(ref c) => c.id(),
            ImpT(ref c) => c.id(),
            ExpT(ref c) => c.id(),
            LitT(ref c) => c.id(),
            LocT(ref c) => c.id(),
            FunT(ref c) => c.id(),
            Attr(ref c) => c.id(),
            CInf(ref c) => c.id(),
            Abst(ref c) => c.id(),
            Dbgi(ref c) => c.id(),
            Docs(ref c) => c.id(),
            Unknown(ref c) => c.id(),
        }
    }
    fn decode_data<R: Read>(id: &Id, reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        use self::StandardChunk::*;
        match id {
            b"Atom" => Ok(Atom(AtomChunk::decode_data(id, reader)?)),
            b"AtU8" => Ok(Atom(AtomChunk::decode_data(id, reader)?)),
            b"Code" => Ok(Code(CodeChunk::decode_data(id, reader)?)),
            b"StrT" => Ok(StrT(StrTChunk::decode_data(id, reader)?)),
            b"ImpT" => Ok(ImpT(ImpTChunk::decode_data(id, reader)?)),
            b"ExpT" => Ok(ExpT(ExpTChunk::decode_data(id, reader)?)),
            b"LitT" => Ok(LitT(LitTChunk::decode_data(id, reader)?)),
            b"LocT" => Ok(LocT(LocTChunk::decode_data(id, reader)?)),
            b"FunT" => Ok(FunT(FunTChunk::decode_data(id, reader)?)),
            b"Attr" => Ok(Attr(AttrChunk::decode_data(id, reader)?)),
            b"CInf" => Ok(CInf(CInfChunk::decode_data(id, reader)?)),
            b"Abst" => Ok(Abst(AbstChunk::decode_data(id, reader)?)),
            b"Dbgi" => Ok(Dbgi(DbgiChunk::decode_data(id, reader)?)),
            b"Docs" => Ok(Docs(DocsChunk::decode_data(id, reader)?)),
            _ => Ok(Unknown(RawChunk::decode_data(id, reader)?)),
        }
    }
    fn encode_data<W: Write>(&self, writer: W) -> Result<()> {
        use self::StandardChunk::*;
        match *self {
            Atom(ref c) => c.encode_data(writer),
            Code(ref c) => c.encode_data(writer),
            StrT(ref c) => c.encode_data(writer),
            ImpT(ref c) => c.encode_data(writer),
            ExpT(ref c) => c.encode_data(writer),
            LitT(ref c) => c.encode_data(writer),
            LocT(ref c) => c.encode_data(writer),
            FunT(ref c) => c.encode_data(writer),
            Attr(ref c) => c.encode_data(writer),
            CInf(ref c) => c.encode_data(writer),
            Abst(ref c) => c.encode_data(writer),
            Dbgi(ref c) => c.encode_data(writer),
            Docs(ref c) => c.encode_data(writer),
            Unknown(ref c) => c.encode_data(writer),
        }
    }
}
