use alloc::rc::Rc;
use alloc::vec::Vec;
use core::fmt;
use core::mem;
use core::str;

use firefly_binary::{BinaryFlags, FromEndianBytes};

use super::{Atom, AtomTable, BinaryData, FunId};
use super::{ByteCode, Function};

pub enum ReadError<T: AtomTable> {
    Eof,
    Magic,
    Invalid,
    InvalidAtom(<T as AtomTable>::AtomError),
}
impl<T: AtomTable> fmt::Debug for ReadError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Eof => write!(f, "unexpected eof"),
            Self::Magic => write!(f, "invalid magic"),
            Self::Invalid => write!(f, "invalid bytecode module"),
            Self::InvalidAtom(err) => write!(f, "invalid atom: {:?}", err),
        }
    }
}
impl<T: AtomTable> From<()> for ReadError<T> {
    fn from(_: ()) -> Self {
        Self::Invalid
    }
}

pub trait Decode<A: Atom, T: AtomTable<Atom = A>>: Sized {
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>>;
}

pub struct BytecodeReader<'a, A, T>
where
    A: Atom,
    T: AtomTable<Atom = A>,
{
    pub(crate) input: &'a [u8],
    atom_rewrites: Vec<A>,
    binary_rewrites: Vec<*const BinaryData>,
    pub(crate) code: ByteCode<A, T>,
    bytes_read: usize,
}
impl<'a, A, T> BytecodeReader<'a, A, T>
where
    A: Atom,
    T: AtomTable<Atom = A> + Default,
{
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            atom_rewrites: Vec::new(),
            binary_rewrites: Vec::new(),
            code: ByteCode::new(),
            bytes_read: 0,
        }
    }
}

impl<'a, A, T> BytecodeReader<'a, A, T>
where
    A: Atom,
    T: AtomTable<Atom = A>,
{
    pub fn read(mut self) -> Result<ByteCode<A, T>, ReadError<T>> {
        self.read_magic()?;

        let num_atoms: usize = self.read_integer()?;
        let num_binaries: usize = self.read_integer()?;
        let num_functions: usize = self.read_integer()?;
        let num_debug_files: usize = self.read_integer()?;
        let num_debug_locations: usize = self.read_integer()?;
        let num_debug_offsets: usize = self.read_integer()?;
        let num_instructions: usize = self.read_integer()?;

        self.read_atoms(num_atoms)?;
        self.read_binaries(num_binaries)?;
        self.read_functions(num_functions)?;
        self.read_debug_info(num_debug_files, num_debug_locations, num_debug_offsets)?;
        self.read_code(num_instructions)?;

        Ok(self.code)
    }

    pub(crate) fn read_magic(&mut self) -> Result<(), ReadError<T>> {
        let magic = self
            .read_bytes(ByteCode::<A, T>::MAGIC.len())
            .map_err(|_| ReadError::Magic)?;
        if magic == ByteCode::<A, T>::MAGIC {
            Ok(())
        } else {
            Err(ReadError::Magic)
        }
    }

    pub(crate) fn read_atoms(&mut self, num_atoms: usize) -> Result<(), ReadError<T>> {
        // Allocate for the rewrites
        self.atom_rewrites.reserve(num_atoms);

        let atom_data_size = mem::size_of::<usize>() * num_atoms;
        let mut atom_data = self.read_bytes(atom_data_size)?;

        for index in 0..num_atoms {
            let (size_bytes, rest) = atom_data.split_at(mem::size_of::<usize>());
            atom_data = rest;
            let atom_size = usize::from_be_bytes(size_bytes.try_into().unwrap());
            let bytes = self.read_bytes(atom_size)?;
            let s = str::from_utf8(bytes).unwrap();
            let atom = self
                .code
                .atoms
                .get_or_insert(s)
                .map_err(ReadError::InvalidAtom)?;
            assert_eq!(index, self.atom_rewrites.len());
            self.atom_rewrites.push(atom);
        }

        Ok(())
    }

    pub(crate) fn read_binaries(&mut self, num_binaries: usize) -> Result<(), ReadError<T>> {
        self.binary_rewrites.reserve(num_binaries);

        let bin_flags_size = mem::size_of::<usize>() * num_binaries;
        let mut flags_data = self.read_bytes(bin_flags_size)?;

        for index in 0..num_binaries {
            let (flag_bytes, rest) = flags_data.split_at(mem::size_of::<usize>());
            flags_data = rest;
            let flags_raw = usize::from_be_bytes(flag_bytes.try_into().unwrap());
            let flags = unsafe { BinaryFlags::from_raw(flags_raw) };
            let bytes = self.read_bytes(flags.size())?;
            let bin = self.code.binaries.get_data_or_insert(flags, bytes);
            assert_eq!(index, self.binary_rewrites.len());
            self.binary_rewrites.push(bin.as_ptr());
        }

        Ok(())
    }

    pub(crate) fn read_functions(&mut self, num_functions: usize) -> Result<(), ReadError<T>> {
        for index in 0..num_functions {
            let mut function = Function::decode(self)?;
            match &mut function {
                Function::Native { ref mut id, .. } => {
                    *id = index as FunId;
                }
                Function::Bif { ref mut id, .. } => {
                    *id = index as FunId;
                }
                Function::Bytecode { ref mut id, .. } => {
                    *id = index as FunId;
                }
            }
            self.code.functions.load(function);
        }

        Ok(())
    }

    pub(crate) fn read_debug_info(
        &mut self,
        num_files: usize,
        num_locs: usize,
        num_offsets: usize,
    ) -> Result<(), ReadError<T>> {
        use super::debuginfo::{FileId, Location, LocationId};

        // Read filenames
        let sizes_size = mem::size_of::<usize>() * num_files;
        let mut size_data = self.read_bytes(sizes_size)?;
        for index in 0..num_files {
            let (size_bytes, rest) = size_data.split_at(mem::size_of::<usize>());
            size_data = rest;
            let size = usize::from_be_bytes(size_bytes.try_into().unwrap());
            let bytes = self.read_bytes(size)?;
            let file: Rc<str> = Rc::from(str::from_utf8(bytes).unwrap());
            self.code.debug_info.files.push(file.clone());
            self.code
                .debug_info
                .files_to_id
                .insert(file, index as FileId);
        }

        // Read locations
        for id in 0..num_locs {
            let file: FileId = self.read_integer()?;
            let line: u32 = self.read_integer()?;
            let column: u32 = self.read_integer()?;

            let loc = Location { file, line, column };
            self.code.debug_info.locations.push(loc);
            self.code
                .debug_info
                .locations_to_id
                .insert(loc, id as LocationId);
        }

        // Read offsets
        for _ in 0..num_offsets {
            let offset: usize = self.read_integer()?;
            let location_id: LocationId = self.read_integer()?;
            assert!((location_id as usize) < self.code.debug_info.locations.len());
            self.code.debug_info.offsets.insert(offset, location_id);
        }

        Ok(())
    }

    pub(crate) fn read_code(&mut self, num_instructions: usize) -> Result<(), ReadError<T>> {
        for _ in 0..num_instructions {
            let op = Decode::decode(self)?;
            self.code.code.push(op);
        }

        Ok(())
    }

    pub fn read_byte(&mut self) -> Result<u8, ReadError<T>> {
        let (byte, rest) = self.input.split_first().ok_or(ReadError::Eof)?;
        self.bytes_read += 1;
        self.input = rest;
        Ok(*byte)
    }

    pub fn read_integer<const N: usize, I: FromEndianBytes<N>>(
        &mut self,
    ) -> Result<I, ReadError<T>> {
        let bytes = self.read_bytes(N)?;
        Ok(I::from_be_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_float(&mut self) -> Result<f64, ReadError<T>> {
        let bytes = self.read_bytes(mem::size_of::<f64>())?;
        Ok(f64::from_be_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], ReadError<T>> {
        if self.input.len() >= len {
            let (bytes, rest) = self.input.split_at(len);
            self.bytes_read += bytes.len();
            self.input = rest;
            Ok(bytes)
        } else {
            Err(ReadError::Eof)
        }
    }

    #[inline]
    pub(super) fn atom_from_offset(&self, offset: usize) -> A {
        self.atom_rewrites[offset]
    }

    #[inline]
    pub(super) fn binary_from_offset(&self, offset: usize) -> *const BinaryData {
        self.binary_rewrites[offset]
    }
}

#[inline(always)]
pub(crate) fn eof_to_invalid<T: AtomTable>(_: ReadError<T>) -> ReadError<T> {
    ReadError::Invalid
}
