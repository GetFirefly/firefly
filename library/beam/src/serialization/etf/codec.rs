mod auxiliary;

use std::io::Write;

use byteorder::BigEndian;
use byteorder::ReadBytesExt;
use byteorder::WriteBytesExt;

use libflate::zlib;

use firefly_number::{BigInt, Int, ToPrimitive};

use self::convert::TryAsRef;

use super::*;

/// Errors which can occur when decoding a term
#[derive(thiserror::Error, Debug)]
pub enum DecodeError {
    #[error("decoding failed: {0}")]
    IO(#[from] std::io::Error),

    #[error("unsupported version: '{version}'")]
    UnsupportedVersion { version: u8 },

    #[error("unknown tag: '{tag}' ")]
    UnknownTag { tag: u8 },

    #[error("unexpected type: {value} is not a {expected}")]
    UnexpectedType { value: Term, expected: String },

    #[error("{value} is out of range {range:?}")]
    OutOfRange {
        value: i32,
        range: std::ops::Range<i32>,
    },
}

/// Errors which can occur when encoding a term
#[derive(thiserror::Error, Debug)]
pub enum EncodeError {
    #[error("encoding failed: {0}")]
    IO(#[from] std::io::Error),

    // a.name.len()
    #[error("atom exceeds maximum byte size: {0}")]
    TooLongAtomName(Atom),

    // i.value.to_bytes_le().1.len()
    #[error("integer is too large, exceeds maximum byte size: {0}")]
    TooLargeInteger(BigInt),

    // r.id.len() * 4
    #[error("reference is too large, exceeds maximum byte size: {0}")]
    TooLargeReferenceId(Reference),
}

pub type DecodeResult = Result<Term, DecodeError>;
pub type EncodeResult = Result<(), EncodeError>;

const VERSION: u8 = 131;
const DISTRIBUTION_HEADER: u8 = 68;
const NEW_FLOAT_EXT: u8 = 70;
const BIT_BINARY_EXT: u8 = 77;
const COMPRESSED_TERM: u8 = 80;
const ATOM_CACHE_REF: u8 = 82;
const SMALL_INTEGER_EXT: u8 = 97;
const INTEGER_EXT: u8 = 98;
const FLOAT_EXT: u8 = 99;
const ATOM_EXT: u8 = 100;
const REFERENCE_EXT: u8 = 101;
const PORT_EXT: u8 = 102;
const PID_EXT: u8 = 103;
const SMALL_TUPLE_EXT: u8 = 104;
const LARGE_TUPLE_EXT: u8 = 105;
const NIL_EXT: u8 = 106;
const STRING_EXT: u8 = 107;
const LIST_EXT: u8 = 108;
const BINARY_EXT: u8 = 109;
const SMALL_BIG_EXT: u8 = 110;
const LARGE_BIG_EXT: u8 = 111;
const NEW_FUN_EXT: u8 = 112;
const EXPORT_EXT: u8 = 113;
const NEW_REFERENCE_EXT: u8 = 114;
const SMALL_ATOM_EXT: u8 = 115;
const MAP_EXT: u8 = 116;
const FUN_EXT: u8 = 117;
const ATOM_UTF8_EXT: u8 = 118;
const SMALL_ATOM_UTF8_EXT: u8 = 119;

pub struct Decoder<R> {
    reader: R,
    buf: Vec<u8>,
}
impl<R: std::io::Read> Decoder<R> {
    pub fn new(reader: R) -> Self {
        Decoder {
            reader,
            buf: Vec::new(),
        }
    }

    pub fn decode(mut self) -> DecodeResult {
        let version = self.reader.read_u8()?;
        if version != VERSION {
            return Err(DecodeError::UnsupportedVersion { version });
        }
        let tag = self.reader.read_u8()?;
        match tag {
            COMPRESSED_TERM => self.decode_compressed_term(),
            DISTRIBUTION_HEADER => unimplemented!(),
            _ => self.decode_term_with_tag(tag),
        }
    }

    fn decode_term(&mut self) -> DecodeResult {
        let tag = self.reader.read_u8()?;
        self.decode_term_with_tag(tag)
    }

    fn decode_term_with_tag(&mut self, tag: u8) -> DecodeResult {
        match tag {
            NEW_FLOAT_EXT => self.decode_new_float_ext(),
            BIT_BINARY_EXT => self.decode_bit_binary_ext(),
            ATOM_CACHE_REF => unimplemented!(),
            SMALL_INTEGER_EXT => self.decode_small_integer_ext(),
            INTEGER_EXT => self.decode_integer_ext(),
            FLOAT_EXT => self.decode_float_ext(),
            ATOM_EXT => self.decode_atom_ext(),
            REFERENCE_EXT => self.decode_reference_ext(),
            PORT_EXT => self.decode_port_ext(),
            PID_EXT => self.decode_pid_ext(),
            SMALL_TUPLE_EXT => self.decode_small_tuple_ext(),
            LARGE_TUPLE_EXT => self.decode_large_tuple_ext(),
            NIL_EXT => self.decode_nil_ext(),
            STRING_EXT => self.decode_string_ext(),
            LIST_EXT => self.decode_list_ext(),
            BINARY_EXT => self.decode_binary_ext(),
            SMALL_BIG_EXT => self.decode_small_big_ext(),
            LARGE_BIG_EXT => self.decode_large_big_ext(),
            NEW_FUN_EXT => self.decode_new_fun_ext(),
            EXPORT_EXT => self.decode_export_ext(),
            NEW_REFERENCE_EXT => self.decode_new_reference_ext(),
            SMALL_ATOM_EXT => self.decode_small_atom_ext(),
            MAP_EXT => self.decode_map_ext(),
            FUN_EXT => self.decode_fun_ext(),
            ATOM_UTF8_EXT => self.decode_atom_utf8_ext(),
            SMALL_ATOM_UTF8_EXT => self.decode_small_atom_utf8_ext(),
            _ => Err(DecodeError::UnknownTag { tag }),
        }
    }

    fn decode_compressed_term(&mut self) -> DecodeResult {
        let _uncompressed_size = self.reader.read_u32::<BigEndian>()? as usize;
        let zlib_decoder = zlib::Decoder::new(&mut self.reader)?;
        let mut decoder = Decoder::new(zlib_decoder);
        decoder.decode_term()
    }

    fn decode_nil_ext(&mut self) -> DecodeResult {
        Ok(Term::from(List::nil()))
    }

    fn decode_string_ext(&mut self) -> DecodeResult {
        let size = self.reader.read_u16::<BigEndian>()? as usize;
        let mut elements = Vec::with_capacity(size);
        for _ in 0..size {
            elements.push(Term::from(Int::from(self.reader.read_u8()? as i64)));
        }
        Ok(Term::from(List::from(elements)))
    }

    fn decode_list_ext(&mut self) -> DecodeResult {
        let count = self.reader.read_u32::<BigEndian>()? as usize;
        let mut elements = Vec::with_capacity(count);
        for _ in 0..count {
            elements.push(self.decode_term()?);
        }
        let last = self.decode_term()?;
        if last
            .try_as_ref()
            .map(|l: &List| l.is_nil())
            .unwrap_or(false)
        {
            Ok(Term::from(List::from(elements)))
        } else {
            Ok(Term::from(ImproperList::from((elements, last))))
        }
    }

    fn decode_small_tuple_ext(&mut self) -> DecodeResult {
        let count = self.reader.read_u8()? as usize;
        let mut elements = Vec::with_capacity(count);
        for _ in 0..count {
            elements.push(self.decode_term()?);
        }
        Ok(Term::from(Tuple::from(elements)))
    }

    fn decode_large_tuple_ext(&mut self) -> DecodeResult {
        let count = self.reader.read_u32::<BigEndian>()? as usize;
        let mut elements = Vec::with_capacity(count);
        for _ in 0..count {
            elements.push(self.decode_term()?);
        }
        Ok(Term::from(Tuple::from(elements)))
    }

    fn decode_map_ext(&mut self) -> DecodeResult {
        let count = self.reader.read_u32::<BigEndian>()? as usize;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let k = self.decode_term()?;
            let v = self.decode_term()?;
            entries.push((k, v));
        }
        Ok(Term::from(Map::from(entries)))
    }

    fn decode_binary_ext(&mut self) -> DecodeResult {
        let size = self.reader.read_u32::<BigEndian>()? as usize;
        let mut buf = vec![0; size];
        self.reader.read_exact(&mut buf)?;
        Ok(Term::from(Binary::from(buf)))
    }

    fn decode_bit_binary_ext(&mut self) -> DecodeResult {
        let size = self.reader.read_u32::<BigEndian>()? as usize;
        let tail_bits_size = self.reader.read_u8()?;
        let mut buf = vec![0; size];
        self.reader.read_exact(&mut buf)?;
        if !buf.is_empty() {
            let last = buf[size - 1] >> (8 - tail_bits_size);
            buf[size - 1] = last;
        }
        Ok(Term::from(BitBinary::from((buf, tail_bits_size))))
    }

    fn decode_pid_ext(&mut self) -> DecodeResult {
        let node = self.decode_term().and_then(auxiliary::term_into_atom)?;
        Ok(Term::from(Pid {
            node,
            id: self.reader.read_u32::<BigEndian>()?,
            serial: self.reader.read_u32::<BigEndian>()?,
            creation: self.reader.read_u8()?,
        }))
    }

    fn decode_port_ext(&mut self) -> DecodeResult {
        let node: Atom = self.decode_term().and_then(|t| {
            t.try_into().map_err(|t| DecodeError::UnexpectedType {
                value: t,
                expected: "Atom".to_string(),
            })
        })?;
        Ok(Term::from(Port {
            node,
            id: self.reader.read_u32::<BigEndian>()?,
            creation: self.reader.read_u8()?,
        }))
    }

    fn decode_reference_ext(&mut self) -> DecodeResult {
        let node = self.decode_term().and_then(auxiliary::term_into_atom)?;
        Ok(Term::from(Reference {
            node,
            id: vec![self.reader.read_u32::<BigEndian>()?],
            creation: self.reader.read_u8()?,
        }))
    }

    fn decode_new_reference_ext(&mut self) -> DecodeResult {
        let id_count = self.reader.read_u16::<BigEndian>()? as usize;
        let node = self.decode_term().and_then(auxiliary::term_into_atom)?;
        let creation = self.reader.read_u8()?;
        let mut id = Vec::with_capacity(id_count);
        for _ in 0..id_count {
            id.push(self.reader.read_u32::<BigEndian>()?);
        }
        Ok(Term::from(Reference { node, id, creation }))
    }

    fn decode_export_ext(&mut self) -> DecodeResult {
        let module = self.decode_term().and_then(auxiliary::term_into_atom)?;
        let function = self.decode_term().and_then(auxiliary::term_into_atom)?;
        let arity =
            self.decode_term()
                .and_then(|t| auxiliary::term_into_ranged_integer(t, 0..0xFF))? as u8;
        Ok(Term::from(ExternalFun {
            module,
            function,
            arity,
        }))
    }

    fn decode_fun_ext(&mut self) -> DecodeResult {
        let num_free = self.reader.read_u32::<BigEndian>()?;
        let pid = self.decode_term().and_then(auxiliary::term_into_pid)?;
        let module = self.decode_term().and_then(auxiliary::term_into_atom)?;
        let index = self.decode_term().and_then(auxiliary::term_into_i32)?;
        let uniq = self.decode_term().and_then(auxiliary::term_into_i32)?;
        let mut vars = Vec::with_capacity(num_free as usize);
        for _ in 0..num_free {
            vars.push(self.decode_term()?);
        }
        Ok(Term::from(InternalFun::Old {
            module,
            pid,
            free_vars: vars,
            index,
            uniq,
        }))
    }

    fn decode_new_fun_ext(&mut self) -> DecodeResult {
        let _size = self.reader.read_u32::<BigEndian>()?;
        let arity = self.reader.read_u8()?;
        let mut uniq = [0; 16];
        self.reader.read_exact(&mut uniq)?;
        let index = self.reader.read_u32::<BigEndian>()?;
        let num_free = self.reader.read_u32::<BigEndian>()?;
        let module = self.decode_term().and_then(auxiliary::term_into_atom)?;
        let old_index = self.decode_term().and_then(auxiliary::term_into_i32)?;
        let old_uniq = self.decode_term().and_then(auxiliary::term_into_i32)?;
        let pid = self.decode_term().and_then(auxiliary::term_into_pid)?;
        let mut vars = Vec::with_capacity(num_free as usize);
        for _ in 0..num_free {
            vars.push(self.decode_term()?);
        }
        Ok(Term::from(InternalFun::New {
            module,
            arity,
            pid,
            free_vars: vars,
            index,
            uniq,
            old_index,
            old_uniq,
        }))
    }

    fn decode_new_float_ext(&mut self) -> DecodeResult {
        let value = self.reader.read_f64::<BigEndian>()?;
        Ok(Term::from(Float::from(value)))
    }

    fn decode_float_ext(&mut self) -> DecodeResult {
        let mut buf = [0; 31];
        self.reader.read_exact(&mut buf)?;
        let float_str = std::str::from_utf8(&mut buf)
            .or_else(|e| auxiliary::invalid_data_error(e.to_string()))?
            .trim_end_matches(0 as char);
        let value = float_str
            .parse::<f32>()
            .or_else(|e| auxiliary::invalid_data_error(e.to_string()))?;
        Ok(Term::from(Float::from(value as f64)))
    }

    fn decode_small_integer_ext(&mut self) -> DecodeResult {
        let value = self.reader.read_u8()?;
        Ok(Term::from(Int::from(value as i64)))
    }

    fn decode_integer_ext(&mut self) -> DecodeResult {
        let value = self.reader.read_i32::<BigEndian>()?;
        Ok(Term::from(Int::from(value)))
    }

    fn decode_small_big_ext(&mut self) -> DecodeResult {
        let count = self.reader.read_u8()? as usize;
        let sign = self.reader.read_u8()?;
        self.buf.resize(count, 0);
        self.reader.read_exact(&mut self.buf)?;
        let value = BigInt::from_bytes_le(auxiliary::byte_to_sign(sign)?, &self.buf);
        Ok(Term::from(Int::from(value)))
    }

    fn decode_large_big_ext(&mut self) -> DecodeResult {
        let count = self.reader.read_u32::<BigEndian>()? as usize;
        let sign = self.reader.read_u8()?;
        self.buf.resize(count, 0);
        self.reader.read_exact(&mut self.buf)?;
        let value = BigInt::from_bytes_le(auxiliary::byte_to_sign(sign)?, &self.buf);
        Ok(Term::from(Int::from(value)))
    }

    fn decode_atom_ext(&mut self) -> DecodeResult {
        let len = self.reader.read_u16::<BigEndian>()?;
        self.buf.resize(len as usize, 0);
        self.reader.read_exact(&mut self.buf)?;
        let name = auxiliary::latin1_bytes_to_string(&self.buf)?;
        Ok(Term::from(Atom {
            name: Symbol::intern(&name),
        }))
    }

    fn decode_small_atom_ext(&mut self) -> DecodeResult {
        let len = self.reader.read_u8()?;
        self.buf.resize(len as usize, 0);
        self.reader.read_exact(&mut self.buf)?;
        let name = auxiliary::latin1_bytes_to_string(&self.buf)?;
        Ok(Term::from(Atom {
            name: Symbol::intern(&name),
        }))
    }

    fn decode_atom_utf8_ext(&mut self) -> DecodeResult {
        let len = self.reader.read_u16::<BigEndian>()?;
        self.buf.resize(len as usize, 0);
        self.reader.read_exact(&mut self.buf)?;
        let name = std::str::from_utf8(&self.buf)
            .or_else(|e| auxiliary::invalid_data_error(e.to_string()))?;
        Ok(Term::from(Atom::from(name)))
    }

    fn decode_small_atom_utf8_ext(&mut self) -> DecodeResult {
        let len = self.reader.read_u8()?;
        self.buf.resize(len as usize, 0);
        self.reader.read_exact(&mut self.buf)?;
        let name = std::str::from_utf8(&self.buf)
            .or_else(|e| auxiliary::invalid_data_error(e.to_string()))?;
        Ok(Term::from(Atom::from(name)))
    }
}

pub struct Encoder<W> {
    writer: W,
}
impl<W: std::io::Write> Encoder<W> {
    pub fn new(writer: W) -> Self {
        Encoder { writer }
    }

    pub fn encode(mut self, term: &Term) -> EncodeResult {
        self.writer.write_u8(VERSION)?;
        self.encode_term(term)
    }

    fn encode_term(&mut self, term: &Term) -> EncodeResult {
        match term {
            Term::Atom(ref x) => self.encode_atom(x),
            Term::String(ref x) => {
                let chars = x
                    .value
                    .as_str()
                    .get()
                    .chars()
                    .map(|c| Term::Integer((c as i64).into()))
                    .collect::<Vec<_>>();
                let list: List = chars.into();
                self.encode_list(&list)
            }
            Term::Integer(ref i) => match i {
                Int::Small(i) => {
                    let i = *i;
                    let j: Option<i32> = i.try_into().ok();
                    match j {
                        Some(i) => self.encode_fix_integer(i),
                        None => {
                            let i = BigInt::from(i);
                            self.encode_big_integer(&i)
                        }
                    }
                }
                Int::Big(ref i) => self.encode_big_integer(i),
            },
            Term::Float(ref x) => self.encode_float(x),
            Term::Pid(ref x) => self.encode_pid(x),
            Term::Port(ref x) => self.encode_port(x),
            Term::Reference(ref x) => self.encode_reference(x),
            Term::ExternalFun(ref x) => self.encode_external_fun(x),
            Term::InternalFun(ref x) => self.encode_internal_fun(x),
            Term::Binary(ref x) => self.encode_binary(x),
            Term::BitBinary(ref x) => self.encode_bit_binary(x),
            Term::List(ref x) => self.encode_list(x),
            Term::ImproperList(ref x) => self.encode_improper_list(x),
            Term::Tuple(ref x) => self.encode_tuple(x),
            Term::Map(ref x) => self.encode_map(x),
        }
    }

    fn encode_nil(&mut self) -> EncodeResult {
        self.writer.write_u8(NIL_EXT)?;
        Ok(())
    }

    fn encode_list(&mut self, x: &List) -> EncodeResult {
        let to_byte = |e: &Term| {
            let i: Option<&Int> = e.try_as_ref();
            i.and_then(|i| i.to_u8())
        };
        if !x.elements.is_empty()
            && x.elements.len() <= std::u16::MAX as usize
            && x.elements.iter().all(|e| to_byte(e).is_some())
        {
            self.writer.write_u8(STRING_EXT)?;
            self.writer
                .write_u16::<BigEndian>(x.elements.len() as u16)?;
            for b in x.elements.iter().map(|e| to_byte(e).unwrap()) {
                self.writer.write_u8(b)?;
            }
        } else {
            if !x.is_nil() {
                self.writer.write_u8(LIST_EXT)?;
                self.writer
                    .write_u32::<BigEndian>(x.elements.len() as u32)?;
                for e in &x.elements {
                    self.encode_term(e)?;
                }
            }
            self.encode_nil()?;
        }
        Ok(())
    }

    fn encode_improper_list(&mut self, x: &ImproperList) -> EncodeResult {
        self.writer.write_u8(LIST_EXT)?;
        self.writer
            .write_u32::<BigEndian>(x.elements.len() as u32)?;
        for e in &x.elements {
            self.encode_term(e)?;
        }
        self.encode_term(&x.last)?;
        Ok(())
    }

    fn encode_tuple(&mut self, x: &Tuple) -> EncodeResult {
        if x.elements.len() < 0x100 {
            self.writer.write_u8(SMALL_TUPLE_EXT)?;
            self.writer.write_u8(x.elements.len() as u8)?;
        } else {
            self.writer.write_u8(LARGE_TUPLE_EXT)?;
            self.writer
                .write_u32::<BigEndian>(x.elements.len() as u32)?;
        }
        for e in &x.elements {
            self.encode_term(e)?;
        }
        Ok(())
    }

    fn encode_map(&mut self, x: &Map) -> EncodeResult {
        self.writer.write_u8(MAP_EXT)?;
        self.writer.write_u32::<BigEndian>(x.entries.len() as u32)?;
        for &(ref k, ref v) in &x.entries {
            self.encode_term(k)?;
            self.encode_term(v)?;
        }
        Ok(())
    }

    fn encode_binary(&mut self, x: &Binary) -> EncodeResult {
        self.writer.write_u8(BINARY_EXT)?;
        self.writer.write_u32::<BigEndian>(x.bytes.len() as u32)?;
        self.writer.write_all(&x.bytes)?;
        Ok(())
    }

    fn encode_bit_binary(&mut self, x: &BitBinary) -> EncodeResult {
        self.writer.write_u8(BIT_BINARY_EXT)?;
        self.writer.write_u32::<BigEndian>(x.bytes.len() as u32)?;
        self.writer.write_u8(x.tail_bits_size)?;
        if !x.bytes.is_empty() {
            self.writer.write_all(&x.bytes[0..x.bytes.len() - 1])?;
            self.writer
                .write_u8(x.bytes[x.bytes.len() - 1] << (8 - x.tail_bits_size))?;
        }
        Ok(())
    }

    fn encode_float(&mut self, x: &Float) -> EncodeResult {
        self.writer.write_u8(NEW_FLOAT_EXT)?;
        self.writer.write_f64::<BigEndian>(x.inner())?;
        Ok(())
    }

    fn encode_atom(&mut self, x: &Atom) -> EncodeResult {
        if x.name.as_str().len() > 0xFFFF {
            return Err(EncodeError::TooLongAtomName(x.clone()));
        }

        let is_ascii = x.name.as_str().as_bytes().iter().all(|&c| c < 0x80);
        if is_ascii {
            self.writer.write_u8(ATOM_EXT)?;
        } else {
            self.writer.write_u8(ATOM_UTF8_EXT)?;
        }
        self.writer
            .write_u16::<BigEndian>(x.name.as_str().len() as u16)?;
        self.writer.write_all(x.name.as_str().as_bytes())?;
        Ok(())
    }

    fn encode_fix_integer(&mut self, x: i32) -> EncodeResult {
        if 0 <= x && x <= std::u8::MAX as i32 {
            self.writer.write_u8(SMALL_INTEGER_EXT)?;
            self.writer.write_u8(x as u8)?;
        } else {
            self.writer.write_u8(INTEGER_EXT)?;
            self.writer.write_i32::<BigEndian>(x)?;
        }
        Ok(())
    }

    fn encode_big_integer(&mut self, x: &BigInt) -> EncodeResult {
        let (sign, bytes) = x.to_bytes_le();
        if bytes.len() <= std::u8::MAX as usize {
            self.writer.write_u8(SMALL_BIG_EXT)?;
            self.writer.write_u8(bytes.len() as u8)?;
        } else if bytes.len() <= std::u32::MAX as usize {
            self.writer.write_u8(LARGE_BIG_EXT)?;
            self.writer.write_u32::<BigEndian>(bytes.len() as u32)?;
        } else {
            return Err(EncodeError::TooLargeInteger(x.clone()));
        }
        self.writer.write_u8(auxiliary::sign_to_byte(sign))?;
        self.writer.write_all(&bytes)?;
        Ok(())
    }

    fn encode_pid(&mut self, x: &Pid) -> EncodeResult {
        self.writer.write_u8(PID_EXT)?;
        self.encode_atom(&x.node)?;
        self.writer.write_u32::<BigEndian>(x.id)?;
        self.writer.write_u32::<BigEndian>(x.serial)?;
        self.writer.write_u8(x.creation)?;
        Ok(())
    }

    fn encode_port(&mut self, x: &Port) -> EncodeResult {
        self.writer.write_u8(PORT_EXT)?;
        self.encode_atom(&x.node)?;
        self.writer.write_u32::<BigEndian>(x.id)?;
        self.writer.write_u8(x.creation)?;
        Ok(())
    }

    fn encode_reference(&mut self, x: &Reference) -> EncodeResult {
        self.writer.write_u8(NEW_REFERENCE_EXT)?;
        if x.id.len() > std::u16::MAX as usize {
            return Err(EncodeError::TooLargeReferenceId(x.clone()));
        }
        self.writer.write_u16::<BigEndian>(x.id.len() as u16)?;
        self.encode_atom(&x.node)?;
        self.writer.write_u8(x.creation)?;
        for n in &x.id {
            self.writer.write_u32::<BigEndian>(*n)?;
        }
        Ok(())
    }

    fn encode_external_fun(&mut self, x: &ExternalFun) -> EncodeResult {
        self.writer.write_u8(EXPORT_EXT)?;
        self.encode_atom(&x.module)?;
        self.encode_atom(&x.function)?;
        self.encode_fix_integer(x.arity as i32)?;
        Ok(())
    }

    fn encode_internal_fun(&mut self, x: &InternalFun) -> EncodeResult {
        match *x {
            InternalFun::Old {
                ref module,
                ref pid,
                ref free_vars,
                index,
                uniq,
            } => {
                self.writer.write_u8(FUN_EXT)?;
                self.writer.write_u32::<BigEndian>(free_vars.len() as u32)?;
                self.encode_pid(pid)?;
                self.encode_atom(module)?;
                self.encode_fix_integer(index)?;
                self.encode_fix_integer(uniq)?;
                for v in free_vars {
                    self.encode_term(v)?;
                }
            }
            InternalFun::New {
                ref module,
                arity,
                ref pid,
                ref free_vars,
                index,
                ref uniq,
                old_index,
                old_uniq,
            } => {
                self.writer.write_u8(NEW_FUN_EXT)?;

                let mut buf = Vec::new();
                {
                    let mut tmp = Encoder::new(&mut buf);
                    tmp.writer.write_u8(arity)?;
                    tmp.writer.write_all(uniq)?;
                    tmp.writer.write_u32::<BigEndian>(index)?;
                    tmp.writer.write_u32::<BigEndian>(free_vars.len() as u32)?;
                    tmp.encode_atom(module)?;
                    tmp.encode_fix_integer(old_index)?;
                    tmp.encode_fix_integer(old_uniq)?;
                    tmp.encode_pid(pid)?;
                    for v in free_vars {
                        tmp.encode_term(v)?;
                    }
                }
                self.writer.write_u32::<BigEndian>(4 + buf.len() as u32)?;
                self.writer.write_all(&buf)?;
            }
        }
        Ok(())
    }
}
