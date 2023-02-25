use alloc::{vec, vec::Vec};
use core::ptr;

use firefly_binary::ToEndianBytes;

use super::{Atom, AtomTable, BinaryData, ByteCode, Function, HashMap, Opcode};

pub trait Encode<A: Atom> {
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()>;
}

pub struct BytecodeWriter<'a, A: Atom> {
    writer: &'a mut dyn std::io::Write,
    atom_offsets: HashMap<A, usize>,
    binary_offsets: HashMap<*const BinaryData, usize>,
    scratch: Vec<u8>,
}
impl<'a, A: Atom> BytecodeWriter<'a, A> {
    pub fn new(writer: &'a mut dyn std::io::Write) -> Self {
        Self {
            writer,
            atom_offsets: HashMap::default(),
            binary_offsets: HashMap::default(),
            scratch: vec![],
        }
    }

    pub fn write<'code, T>(mut self, code: &'code ByteCode<A, T>) -> std::io::Result<()>
    where
        T: AtomTable<Atom = A>,
    {
        self.write_header(code)?;
        self.write_atoms(code)?;
        self.write_binaries(code)?;
        self.write_functions(code)?;
        self.write_debug_info(code)?;
        self.write_code(code)
    }

    pub(crate) fn write_header<'code, T>(
        &mut self,
        code: &'code ByteCode<A, T>,
    ) -> std::io::Result<()>
    where
        T: AtomTable<Atom = A>,
    {
        // 1. Write magic
        self.writer.write_all(ByteCode::<A, T>::MAGIC)?;
        // 2. Write size of atom table (number of atoms)
        self.writer.write_all(&code.atoms.len().to_be_bytes())?;
        // 3. Write size of binaries table (number of binaries)
        self.writer.write_all(&code.binaries.len().to_be_bytes())?;
        // 4. Write size of functions table (number of functions)
        self.writer.write_all(&code.functions.len().to_be_bytes())?;
        // 5. Write size of debug info (number of files, number of locations, number of offsets)
        self.writer
            .write_all(&code.debug_info.files.len().to_be_bytes())?;
        self.writer
            .write_all(&code.debug_info.locations.len().to_be_bytes())?;
        self.writer
            .write_all(&code.debug_info.offsets.len().to_be_bytes())?;
        // 6. Write size of code (number of instructions)
        self.writer.write_all(&code.code.len().to_be_bytes())
    }

    pub(crate) fn write_atoms<'code, T>(
        &mut self,
        code: &'code ByteCode<A, T>,
    ) -> std::io::Result<()>
    where
        T: AtomTable<Atom = A>,
    {
        self.scratch.clear();

        for (index, atom) in code.atoms.iter().enumerate() {
            let (_, atom_size) = A::into_raw_parts(A::unpack(atom));
            let atom_bytes = atom.as_bytes();
            self.scratch.extend_from_slice(atom_bytes);
            debug_assert_eq!(atom_bytes.len(), atom_size);
            self.write_integer(atom_size)?;
            self.atom_offsets.insert(atom, index);
        }

        self.writer.write_all(self.scratch.as_slice())
    }

    pub(crate) fn write_binaries<'code, T>(
        &mut self,
        code: &'code ByteCode<A, T>,
    ) -> std::io::Result<()>
    where
        T: AtomTable<Atom = A>,
    {
        self.scratch.clear();

        for (index, data) in code.binaries.iter().enumerate() {
            let bin = unsafe { data.as_ref() };
            let flags = bin.flags();
            self.scratch.extend_from_slice(bin.as_bytes());
            self.write_integer(flags.into_raw())?;
            self.binary_offsets.insert(bin, index);
        }

        // Append the raw data bytes
        self.writer.write_all(self.scratch.as_slice())
    }

    pub(crate) fn write_functions<'code, T>(
        &mut self,
        code: &'code ByteCode<A, T>,
    ) -> std::io::Result<()>
    where
        T: AtomTable<Atom = A>,
    {
        // Since `Function` is variable-length, we simply encode each function after the next
        //
        // Since the functions are always in the same order, FunIds remain unchanged, so
        // no instruction fixups are required
        for function in code.functions.iter() {
            Function::encode(function, self)?;
        }

        Ok(())
    }

    pub(crate) fn write_debug_info<'code, T>(
        &mut self,
        code: &'code ByteCode<A, T>,
    ) -> std::io::Result<()>
    where
        T: AtomTable<Atom = A>,
    {
        // Write files
        self.scratch.clear();

        for file in code.debug_info.files.iter() {
            let bytes = file.as_bytes();
            self.scratch.extend_from_slice(bytes);
            self.write_integer(bytes.len())?;
        }
        self.writer.write_all(self.scratch.as_slice())?;

        // Write locations
        for loc in code.debug_info.locations.iter() {
            self.write_integer(loc.file)?;
            self.write_integer(loc.line)?;
            self.write_integer(loc.column)?;
        }

        // Write offsets
        for (offset, location_id) in code.debug_info.offsets.iter() {
            self.write_integer(*offset)?;
            self.write_integer(*location_id)?;
        }

        Ok(())
    }

    pub(crate) fn write_code<'code, T>(
        &mut self,
        code: &'code ByteCode<A, T>,
    ) -> std::io::Result<()>
    where
        T: AtomTable<Atom = A>,
    {
        use super::ops::{LoadAtom, LoadBinary, LoadBitstring};

        // Instruction encoding is delegated to Opcode
        for op in code.code.iter() {
            match op {
                Opcode::LoadAtom(LoadAtom { dest, value }) => {
                    let offset = self.atom_offsets[value];
                    let value = A::pack(unsafe { A::from_raw_parts(offset as *const u8, 0) });
                    let op = Opcode::LoadAtom(LoadAtom { dest: *dest, value });
                    op.encode(self)?;
                }
                Opcode::LoadBinary(LoadBinary { dest, value }) => {
                    let offset = self.binary_offsets[value];
                    let value = ptr::from_raw_parts(offset as *const (), ptr::metadata(*value));
                    let op = Opcode::LoadBinary(LoadBinary { dest: *dest, value });
                    op.encode(self)?;
                }
                Opcode::LoadBitstring(LoadBitstring { dest, value }) => {
                    let offset = self.binary_offsets[value];
                    let value = ptr::from_raw_parts(offset as *const (), ptr::metadata(*value));
                    let op = Opcode::LoadBitstring(LoadBitstring { dest: *dest, value });
                    op.encode(self)?;
                }
                op => {
                    op.encode(self)?;
                }
            }
        }

        Ok(())
    }

    #[inline]
    pub fn write_byte(&mut self, byte: u8) -> std::io::Result<()> {
        self.writer.write_all(&[byte])
    }

    #[inline]
    pub fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.writer.write_all(bytes)
    }

    #[inline]
    pub fn write_integer<const N: usize, I: ToEndianBytes<N>>(
        &mut self,
        value: I,
    ) -> std::io::Result<()> {
        self.writer.write_all(&value.to_be_bytes())
    }

    #[inline]
    pub fn write_float(&mut self, value: f64) -> std::io::Result<()> {
        self.writer.write_all(&value.to_be_bytes())
    }

    #[inline]
    pub fn write_atom(&mut self, atom: A) -> std::io::Result<()> {
        let offset: usize = self.atom_offsets[&atom];
        self.write_integer(offset)
    }
}
