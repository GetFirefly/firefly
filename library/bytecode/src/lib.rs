#![allow(incomplete_features)]
#![no_std]
#![feature(maybe_uninit_as_bytes)]
#![feature(maybe_uninit_slice)]
#![feature(new_uninit)]
#![feature(ptr_metadata)]
#![feature(layout_for_ptr)]
#![feature(allocator_api)]
#![feature(core_intrinsics)]
#![feature(assert_matches)]
#![feature(slice_as_chunks)]
#![feature(test)]
#![feature(return_position_impl_trait_in_trait)]

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;
#[cfg(test)]
extern crate test;

mod atoms;
mod builder;
mod debuginfo;
pub mod ops;
mod reader;
#[cfg(test)]
mod tests;
mod text;
#[cfg(any(test, feature = "std"))]
mod writer;

pub use self::atoms::{Atom, AtomTable, AtomicStr};
pub use self::builder::{Block, BlockId, Builder, FunctionBuilder};
pub use self::debuginfo::{DebugInfoTable, FileId, Location, LocationId, SourceLocation, Symbol};
pub use self::ops::Opcode;
pub use self::reader::{BytecodeReader, ReadError};
#[cfg(any(test, feature = "std"))]
pub use self::writer::BytecodeWriter;

use alloc::alloc::Layout;
use alloc::collections::btree_map::BTreeMap;
use alloc::{vec, vec::Vec};
use core::assert_matches::assert_matches;
use core::fmt;
use core::mem;
use core::ptr::{self, NonNull};
use core::str;

use firefly_arena::DroplessArena;
use firefly_binary::{BinaryFlags, Encoding};

pub type Register = u16;
pub type Immediate = i16;
pub type JumpOffset = i16;
pub type Arity = u8;
pub type FunId = u16;
pub type LiteralId = u16;
pub type DataOffset = usize;

pub(crate) type HashMap<K, V> =
    hashbrown::HashMap<K, V, core::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

/// This is equivalent in representation to `firefly_rt::term::BinaryData`
///
/// We use this to ensure the data layout in memory is the same, without having to
/// depend on the `firefly_rt` crate directly
#[repr(C, align(16))]
pub struct BinaryData {
    /// * `0111111111111xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxTTTT11` = Header (canonical
    ///   NaN + 0x03 + 4-bit tag + 45-bit arity val)
    pub header: u64,
    pub data: [u8],
}
impl PartialEq for BinaryData {
    fn eq(&self, other: &Self) -> bool {
        self.header == other.header && self.as_bytes() == other.as_bytes()
    }
}
impl fmt::Debug for BinaryData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let bytes = self.as_bytes();
        if self.flags().is_raw() {
            firefly_binary::helpers::display_bytes(bytes.iter().copied(), f)
        } else {
            match core::str::from_utf8(bytes) {
                Ok(s) => f.write_str(s),
                Err(_) => firefly_binary::helpers::display_bytes(bytes.iter().copied(), f),
            }
        }
    }
}
impl fmt::Display for BinaryData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let bytes = self.as_bytes();
        if self.flags().is_raw() {
            firefly_binary::helpers::display_bytes(bytes.iter().copied(), f)
        } else {
            match core::str::from_utf8(bytes) {
                Ok(s) => f.write_str(s),
                Err(_) => firefly_binary::helpers::display_bytes(bytes.iter().copied(), f),
            }
        }
    }
}
impl BinaryData {
    const NAN: u64 = unsafe { mem::transmute::<f64, u64>(f64::NAN) };
    // See library/rt/src/term/header.rs, Tag::Binary
    const BINARY_TAG: u64 = Self::NAN | 0b011111;

    #[inline]
    fn flags(&self) -> BinaryFlags {
        let arity = (self.header & !Self::BINARY_TAG) >> 6;
        unsafe { BinaryFlags::from_raw(arity as usize) }
    }

    #[inline]
    fn set_flags(&mut self, flags: BinaryFlags) {
        let arity = (BinaryFlags::into_raw(flags) as u64) << 6;
        self.header = Self::BINARY_TAG | arity;
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline(always)]
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    fn copy_from_slice(&mut self, bytes: &[u8]) {
        assert_eq!(self.len(), bytes.len());
        self.data.copy_from_slice(bytes)
    }
}

/// This is equivalent in representation to `firefly_rt::function::ModuleFunctionArity`
///
/// We use this to ensure the data layout in memory is the same, without having to
/// depend on the `firefly_rt` crate directly
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct ModuleFunctionArity<A: Atom> {
    pub module: A,
    pub function: A,
    pub arity: u8,
}
impl<A: Atom> fmt::Debug for ModuleFunctionArity<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}/{}", &self.module, &self.function, &self.arity)
    }
}
impl<A: Atom> fmt::Display for ModuleFunctionArity<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}/{}", &self.module, &self.function, &self.arity)
    }
}

/// Used to represent the valid atoms for `erlang:raise/3`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum ErrorKind {
    Throw = 0,
    Error,
    Exit,
}
impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Throw => f.write_str("throw"),
            Self::Error => f.write_str("error"),
            Self::Exit => f.write_str("exit"),
        }
    }
}

/// Represents the types of functions which can be defined/referenced in bytecode
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Function<A: Atom> {
    /// A natively-implemented function using the C calling convention
    Native {
        id: FunId,
        /// The symbol name of the native function which must be exported in any
        /// executable which loads this bytecode
        name: A,
        /// The arity of the callee
        arity: u8,
    },
    /// An Erlang built-in, i.e. a natively-implemented function using the Erlang calling
    /// convention
    Bif {
        id: FunId,
        /// When converted to a string, this should be the name of a native function
        /// which is exported in the executable which loads this bytecode
        mfa: ModuleFunctionArity<A>,
    },
    /// An Erlang function implemented in bytecode
    Bytecode {
        id: FunId,
        is_nif: bool,
        mfa: ModuleFunctionArity<A>,
        offset: usize,
        frame_size: usize,
    },
}
#[cfg(any(test, feature = "std"))]
impl<A: Atom> writer::Encode<A> for Function<A> {
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        match self {
            Self::Native {
                id: _id,
                name,
                arity,
            } => {
                writer.write_byte(0)?;
                writer.write_atom(*name)?;
                writer.write_byte(*arity)
            }
            Self::Bif { id: _id, mfa } => {
                writer.write_byte(1)?;
                writer.write_atom(mfa.module)?;
                writer.write_atom(mfa.function)?;
                writer.write_byte(mfa.arity)
            }
            Self::Bytecode {
                id: _id,
                is_nif,
                mfa,
                offset,
                frame_size,
            } => {
                writer.write_byte(2)?;
                writer.write_byte(*is_nif as u8)?;
                writer.write_atom(mfa.module)?;
                writer.write_atom(mfa.function)?;
                writer.write_byte(mfa.arity)?;
                writer.write_integer(*offset)?;
                writer.write_integer(*frame_size)
            }
        }
    }
}
impl<A: Atom, T: AtomTable<Atom = A>> reader::Decode<A, T> for Function<A> {
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        let tag = reader.read_byte().map_err(reader::eof_to_invalid)?;
        match tag {
            0 => {
                let name_offset = reader.read_integer().map_err(reader::eof_to_invalid)?;
                let name = reader.atom_from_offset(name_offset);
                let arity = reader.read_byte().map_err(reader::eof_to_invalid)?;
                Ok(Self::Native { id: 0, name, arity })
            }
            1 => {
                let module_offset = reader.read_integer().map_err(reader::eof_to_invalid)?;
                let module = reader.atom_from_offset(module_offset);
                let function_offset = reader.read_integer().map_err(reader::eof_to_invalid)?;
                let function = reader.atom_from_offset(function_offset);
                let arity = reader.read_byte().map_err(reader::eof_to_invalid)?;
                let mfa = ModuleFunctionArity {
                    module,
                    function,
                    arity,
                };
                Ok(Self::Bif { id: 0, mfa })
            }
            2 => {
                let is_nif = reader.read_byte().map_err(reader::eof_to_invalid)? != 0;
                let module_offset = reader.read_integer().map_err(reader::eof_to_invalid)?;
                let module = reader.atom_from_offset(module_offset);
                let function_offset = reader.read_integer().map_err(reader::eof_to_invalid)?;
                let function = reader.atom_from_offset(function_offset);
                let arity = reader.read_byte().map_err(reader::eof_to_invalid)?;
                let offset = reader.read_integer().map_err(reader::eof_to_invalid)?;
                let frame_size = reader.read_integer().map_err(reader::eof_to_invalid)?;
                let mfa = ModuleFunctionArity {
                    module,
                    function,
                    arity,
                };
                Ok(Self::Bytecode {
                    id: 0,
                    is_nif,
                    mfa,
                    offset,
                    frame_size,
                })
            }
            _ => Err(ReadError::Invalid),
        }
    }
}
impl<A: Atom> Function<A> {
    #[inline]
    pub fn id(&self) -> FunId {
        match self {
            Self::Native { id, .. } | Self::Bif { id, .. } | Self::Bytecode { id, .. } => *id,
        }
    }

    /// Returns the [`ModuleFunctionArity`] for this function, if it is an Erlang function (built-in
    /// or bytecoded)
    #[inline]
    pub fn mfa(&self) -> Option<&ModuleFunctionArity<A>> {
        match self {
            Self::Bytecode { mfa, .. } | Self::Bif { mfa, .. } => Some(mfa),
            _ => None,
        }
    }

    #[inline]
    pub fn arity(&self) -> usize {
        match self {
            Self::Bytecode { mfa, .. } | Self::Bif { mfa, .. } => mfa.arity as usize,
            Self::Native { arity, .. } => (*arity) as usize,
        }
    }

    /// Obtains the instruction offset from this function, if this is a bytecode function
    #[inline]
    pub fn offset(&self) -> Option<usize> {
        match self {
            Self::Bytecode { offset, .. } => Some(*offset),
            _ => None,
        }
    }

    #[inline]
    pub fn frame_size(&self) -> Option<usize> {
        match self {
            Self::Bytecode { frame_size, .. } => Some(*frame_size),
            _ => None,
        }
    }
}

/// This is a type alias for a `ByteCode` module which is fully self-contained,
/// i.e. it has no references to a global atom table or anything else not held
/// within the module itself.
///
/// This type is primarily used during compilation and testing.
pub type StandardByteCode = ByteCode<AtomicStr, LocalAtomTable>;

#[derive(Debug)]
pub enum InvalidBytecodeError<A: Atom> {
    /// This is raised when validating a bytecode module and a bytecode function definition is
    /// found that has no offset, indicating that it has no body. Such function definitions are
    /// valid during construction, but invalid in a finalized bytecode module.
    IncompleteFunction(ModuleFunctionArity<A>),
    /// An attempt to define a function body twice was made.
    DuplicateDefinition(ModuleFunctionArity<A>),
}

/// This structure represents a single translation unit of bytecode, i.e. it
/// can represent one or more modules and their functions, along with their literals
/// and (optionally) debug information.
///
/// Currently this is designed to contain an entire program's worth of bytecode, unlike
/// BEAM which separates each module into its own bytecode file. We do this because our
/// needs are simpler, and we can be more efficient by keeping everything in one translation
/// unit; but in the future we may want to rework this to allow dynamically reloading certain
/// definitions, or extending the bytecode to add new definitions - this is not supported right
/// now.
pub struct ByteCode<A: Atom, T: AtomTable<Atom = A>> {
    /// A local atom table, containing all of the atoms referenced in this translation unit
    ///
    /// When loading the code, this atom table will be bulk-loaded into the global runtime
    /// atom table, with all of the instructions patched-up accordingly
    pub atoms: T,
    /// A table containing literal binaries referenced by code in this translation unit
    ///
    /// When loading the code, the arena in this table will be "leaked" so that references
    /// to data inside it are never invalidated
    pub binaries: BinaryTable,
    /// A table containing metadata about functions in this translation unit, i.e. what
    /// type of function is it, the instruction offset of the function entry, etc.
    pub functions: FunctionTable<A>,
    /// A table containing debug information which allows reconstructing file/line/column
    /// information in stack traces, or interrogating the same about the current instruction
    /// pointer
    pub debug_info: DebugInfoTable,
    /// The decoded instructions of this translation unit
    ///
    /// The instruction pointer of the emulator running this bytecode is an index into this vector.
    pub code: Vec<Opcode<A>>,
}
unsafe impl<A: Atom, T: AtomTable<Atom = A>> Send for ByteCode<A, T> {}
unsafe impl<A: Atom, T: AtomTable<Atom = A>> Sync for ByteCode<A, T> {}
impl<A: Atom, T: AtomTable<Atom = A> + Default> Default for ByteCode<A, T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
impl<A: Atom, T: AtomTable<Atom = A>> fmt::Display for ByteCode<A, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        text::write(f, self)
    }
}
impl<A: Atom, T: AtomTable<Atom = A> + Default> ByteCode<A, T> {
    /// Create a new, empty bytecode container for use with a [`BytecodeBuilder`]
    pub fn new() -> Self {
        Self {
            atoms: T::default(),
            binaries: BinaryTable::default(),
            functions: FunctionTable::default(),
            debug_info: DebugInfoTable::default(),
            code: vec![],
        }
    }
}
impl<A: Atom, T: AtomTable<Atom = A>> ByteCode<A, T> {
    /// Used to validate that we're loading valid data when loading from raw binary data
    pub const MAGIC: &'static [u8] = b"FIREFLY_";

    pub fn validate(&self) -> Result<(), InvalidBytecodeError<A>> {
        for function in self.functions.iter() {
            match function {
                Function::Bytecode { offset, mfa, .. } => {
                    if *offset == 0 {
                        return Err(InvalidBytecodeError::IncompleteFunction(*mfa));
                    }
                }
                _ => continue,
            }
        }

        Ok(())
    }

    /// This function is used to link two or more bytecode modules together into a single module.
    ///
    /// Specifically, all of the given modules will be merged into `self`, and then a single pass
    /// will be made which:
    ///
    /// * Resolves any static calls which were previously unresolved, converting them to jumps
    /// * Rewrites all offsets to their new location in the merged module
    /// * Rewrites all atom/binary references to use the tables in `self`
    ///
    /// Validation ensures that no function definition conflicts with those defined in other modules
    pub fn link(&mut self, mut other: Vec<Self>) -> Result<(), InvalidBytecodeError<A>> {
        use self::ops::{Call, CallNative, CallStatic, Enter, EnterNative, EnterStatic};
        use self::ops::{Closure, FuncInfo, LoadAtom, LoadBinary, LoadBitstring};

        struct ModuleMap {
            range: core::ops::Range<usize>,
            index: usize,
        }

        let start = self.code.len();
        let mut module_map = Vec::with_capacity(other.len());
        for (i, module) in other.iter_mut().enumerate() {
            if module.code.is_empty() {
                continue;
            }
            let current_ix_offset = self.code.len();
            module_map.push(ModuleMap {
                range: current_ix_offset..(current_ix_offset + module.code.len()),
                index: i,
            });
            for atom in module.atoms.iter() {
                let s = unsafe { str::from_utf8_unchecked(atom.as_bytes()) };
                self.atoms.get_or_insert(s).unwrap();
            }
            self.binaries.append(&module.binaries);
            self.functions
                .append(&module.functions, current_ix_offset)?;
            self.debug_info
                .append(&module.debug_info, current_ix_offset);
            self.code.append(&mut module.code);
        }

        // Split the code into two regions; those which do not need rewriting, and those that do
        let (original, rest) = self.code.split_at_mut(start);
        // For all instructions in the original code, we're only looking for call instructions
        // which should now have known jump targets. This is an optimization.
        for op in original.iter_mut() {
            match op {
                Opcode::CallStatic(CallStatic { callee, dest }) => {
                    let fun = self.functions.get(*callee);
                    match fun {
                        Function::Bytecode {
                            is_nif: false,
                            offset,
                            ..
                        } if *offset > 0 => {
                            *op = Opcode::Call(Call {
                                dest: *dest,
                                offset: *offset,
                            });
                        }
                        Function::Bytecode { mfa, .. } => {
                            return Err(InvalidBytecodeError::IncompleteFunction(*mfa));
                        }
                        Function::Native { arity, .. } => {
                            let callee = *callee;
                            *op = Opcode::CallNative(CallNative {
                                dest: *dest,
                                callee: callee as usize as *const (),
                                arity: *arity,
                            });
                        }
                        _ => continue,
                    }
                }
                Opcode::EnterStatic(EnterStatic { callee }) => {
                    let fun = self.functions.get(*callee);
                    match fun {
                        Function::Bytecode {
                            is_nif: false,
                            offset,
                            ..
                        } if *offset > 0 => {
                            *op = Opcode::Enter(Enter { offset: *offset });
                        }
                        Function::Bytecode { mfa, .. } => {
                            return Err(InvalidBytecodeError::IncompleteFunction(*mfa));
                        }
                        Function::Native { arity, .. } => {
                            let callee = *callee;
                            *op = Opcode::EnterNative(EnterNative {
                                callee: callee as usize as *const (),
                                arity: *arity,
                            });
                        }
                        _ => continue,
                    }
                }
                _ => continue,
            }
        }
        // For the remaining instructions, we must visit every instruction with an
        // absolute offset and rewrite the offset to its new location in the module.
        //
        // In addition to function offsets, we also need to rewrite the entries for atom
        // and binary/bitstring literals.
        let remap_callee = |callee: FunId, module_index: usize| {
            let module = &other[module_index];
            self.functions
                .find_by_mfa(module.functions.get(callee).mfa().unwrap())
                .unwrap()
                .id()
        };
        let mut module_map_iter = module_map.iter();
        let mut current_module = module_map_iter.next().unwrap();
        for (i, op) in rest.iter_mut().enumerate() {
            if i >= current_module.range.end {
                current_module = module_map_iter.next().unwrap();
            }
            match op {
                Opcode::Call(Call { ref mut offset, .. })
                | Opcode::Enter(Enter { ref mut offset, .. }) => {
                    let original_module = &other[current_module.index];
                    let callee = original_module.functions.locate(*offset);
                    let new_offset = self
                        .functions
                        .find_by_mfa(callee.mfa().unwrap())
                        .unwrap()
                        .offset()
                        .unwrap();
                    *offset = new_offset;
                }
                Opcode::CallNative(CallNative { ref mut callee, .. }) => {
                    let new_callee =
                        remap_callee((*callee) as usize as FunId, current_module.index);
                    assert_matches!(self.functions.get(new_callee), Function::Native { .. });
                    *callee = new_callee as usize as *const ();
                }
                Opcode::EnterNative(EnterNative { ref mut callee, .. }) => {
                    let new_callee =
                        remap_callee((*callee) as usize as FunId, current_module.index);
                    assert_matches!(self.functions.get(new_callee), Function::Native { .. });
                    *callee = new_callee as usize as *const ();
                }
                Opcode::FuncInfo(FuncInfo { ref mut id, .. }) => {
                    *id = remap_callee(*id, current_module.index);
                }
                Opcode::CallStatic(CallStatic {
                    ref mut callee,
                    dest,
                }) => {
                    let new_callee = remap_callee(*callee, current_module.index);
                    match self.functions.get(new_callee) {
                        Function::Bytecode {
                            is_nif: false,
                            offset,
                            ..
                        } if *offset > 0 => {
                            *op = Opcode::Call(Call {
                                dest: *dest,
                                offset: *offset,
                            });
                        }
                        Function::Bytecode { mfa, .. } => {
                            return Err(InvalidBytecodeError::IncompleteFunction(*mfa));
                        }
                        Function::Native { arity, .. } => {
                            *op = Opcode::CallNative(CallNative {
                                dest: *dest,
                                callee: new_callee as usize as *const (),
                                arity: *arity,
                            });
                        }
                        _ => {
                            *callee = new_callee;
                        }
                    }
                }
                Opcode::EnterStatic(EnterStatic { ref mut callee }) => {
                    let new_callee = remap_callee(*callee, current_module.index);
                    match self.functions.get(new_callee) {
                        Function::Bytecode {
                            is_nif: false,
                            offset,
                            ..
                        } if *offset > 0 => {
                            *op = Opcode::Enter(Enter { offset: *offset });
                        }
                        Function::Bytecode { mfa, .. } => {
                            return Err(InvalidBytecodeError::IncompleteFunction(*mfa));
                        }
                        Function::Native { arity, .. } => {
                            *op = Opcode::EnterNative(EnterNative {
                                callee: new_callee as usize as *const (),
                                arity: *arity,
                            });
                        }
                        _ => {
                            *callee = new_callee;
                        }
                    }
                }
                Opcode::Closure(Closure {
                    ref mut function, ..
                }) => {
                    let new_function = remap_callee(*function, current_module.index);
                    *function = new_function;
                }
                Opcode::LoadAtom(LoadAtom { ref mut value, .. }) => {
                    let name = unsafe { str::from_utf8_unchecked(value.as_bytes()) };
                    let new_value = self.atoms.get_or_insert(name).unwrap();
                    *value = new_value;
                }
                Opcode::LoadBinary(LoadBinary { ref mut value, .. })
                | Opcode::LoadBitstring(LoadBitstring { ref mut value, .. }) => {
                    let bin = unsafe { &**value };
                    let new_bin = self
                        .binaries
                        .get_data_or_insert(bin.flags(), bin.as_bytes());
                    *value = new_bin.as_ptr();
                }
                _ => continue,
            }
        }

        Ok(())
    }

    /// Inserts an atom into this bytecode's atom table
    ///
    /// This is the only way to get a valid [`Atom`] for use in building a bytecode module
    pub fn insert_atom(&mut self, name: &str) -> Result<A, <T as AtomTable>::AtomError> {
        self.atoms.get_or_insert(name)
    }

    /// Inserts binary data into this bytecode's binary table
    ///
    /// This is the only way to get a valid [`BinaryData`] pointer for use in building a bytecode
    /// module
    pub fn insert_binary(&mut self, bytes: &[u8], encoding: Encoding) -> *const BinaryData {
        let data = self
            .binaries
            .get_data_or_insert(BinaryFlags::new(bytes.len(), encoding), bytes);

        data.as_ptr()
    }

    /// Inserts bitstring data into this bytecode's binary table
    ///
    /// This is the only way to get a valid [`BinaryData`] pointer for use in building a bytecode
    /// module
    pub fn insert_bitstring(&mut self, bytes: &[u8], trailing_bits: usize) -> *const BinaryData {
        let flags = BinaryFlags::new(bytes.len(), Encoding::Raw).with_trailing_bits(trailing_bits);
        let data = self.binaries.get_data_or_insert(flags, bytes);

        data.as_ptr()
    }

    /// Defines a new bytecode function with the given MFA.
    ///
    /// This function will return an error if an attempt is made to define a function more than
    /// once, or an attempt is made to define a function which has the same signature as a BIF.
    pub fn define_function(
        &mut self,
        mfa: ModuleFunctionArity<A>,
        offset: usize,
    ) -> Result<FunId, InvalidBytecodeError<A>> {
        self.functions.define(mfa, offset)
    }

    /// Gets the [`FunId`] assigned to function `mfa`, or defines a new bytecode function with that
    /// name.
    ///
    /// If you want to make sure the function doesn't exist yet, use `function_by_mfa` to look it up
    /// first.
    pub fn get_or_define_function(&mut self, mfa: ModuleFunctionArity<A>) -> FunId {
        self.functions.get_or_define(mfa)
    }

    /// Same as `get_or_define_function`, but for built-in functions that don't have bytecode
    /// instructions.
    pub fn get_or_define_bif(&mut self, mfa: ModuleFunctionArity<A>) -> FunId {
        self.functions.get_or_define_bif(mfa)
    }

    /// Same as `get_or_define_function`, but for natively-implemented functions
    pub fn get_or_define_nif<S: AsRef<str>>(&mut self, name: S, arity: u8) -> FunId {
        let name = self.insert_atom(name.as_ref()).unwrap();
        self.functions.get_or_define_nif(name, arity)
    }

    #[inline]
    pub(crate) fn function_mut(&mut self, id: FunId) -> &mut Function<A> {
        self.functions.get_mut(id)
    }

    /// Look up a function by [`ModuleFunctionArity`]
    ///
    /// For native functions, use `function_by_name`.
    #[inline]
    pub fn function_by_mfa(&self, mfa: &ModuleFunctionArity<A>) -> Option<&Function<A>> {
        self.functions.find_by_mfa(mfa)
    }

    /// Look up a function by name
    ///
    /// This will return matches against any type of function, but is slower as it must
    /// perform string matching.
    #[inline]
    pub fn function_by_name<S: AsRef<str>>(&self, name: S) -> Option<&Function<A>> {
        let name = name.as_ref();
        // Manufacture an atom for the search
        let bytes = name.as_bytes();
        let name_atom = A::pack(unsafe { A::from_raw_parts(bytes.as_ptr(), bytes.len()) });
        // See if we have it by name
        if let Some(fun) = self.functions.find_by_name(name_atom) {
            return Some(fun);
        }
        // We don't, so check and see if this is a valid MFA format and try that
        let (m, rest) = name.split_once(':')?;
        let (f, a) = rest.split_once('/')?;
        let arity: u8 = a.parse().ok()?;
        // Manufacture the atoms for the search
        let module_bytes = m.as_bytes();
        let module =
            unsafe { A::pack(A::from_raw_parts(module_bytes.as_ptr(), module_bytes.len())) };
        let function_bytes = f.as_bytes();
        let function = unsafe {
            A::pack(A::from_raw_parts(
                function_bytes.as_ptr(),
                function_bytes.len(),
            ))
        };
        let mfa = ModuleFunctionArity {
            module,
            function,
            arity,
        };
        self.functions.find_by_mfa(&mfa)
    }

    /// Look up a bytecode function by instruction pointer
    #[inline]
    pub fn function_by_ip(&self, ip: usize) -> &Function<A> {
        self.functions.locate(ip)
    }

    /// Look up a function by [`FunId`]
    #[inline]
    pub fn function_by_id(&self, id: FunId) -> &Function<A> {
        self.functions.get(id)
    }

    /// Look up the instruction offset of a bytecode function by [`FunId`]
    ///
    /// # SAFETY
    ///
    /// This function is not safe to call unless you can guarantee that `id` is the
    /// id of a bytecode function. This is intended for use by an emulator which can
    /// make that guarantee when calling this.
    ///
    /// You should look up a function and then ask for its offset, rather than calling
    /// this blindly.
    #[inline(always)]
    pub unsafe fn function_offset(&self, id: FunId) -> usize {
        self.functions.offset(id)
    }

    /// Returns the instruction offset of the last instruction in this bytecode module
    #[inline(always)]
    pub fn last_instruction(&self) -> usize {
        self.code.len() - 1
    }

    /// Returns the next instruction offset in this bytecode module
    #[inline(always)]
    pub(crate) fn next_instruction(&self) -> usize {
        self.code.len()
    }

    /// Attempts to symbolicate the given instruction offset in this bytecode module, by
    /// looking up the function it is defined in, and looking for the nearest debuginfo for that
    /// instruction.
    ///
    /// This function always returns `None` for natively-implemented functions.
    ///
    /// NOTE: This function will panic in debug mode if the instruction pointer given is out of
    /// bounds
    pub fn instruction_symbol(&self, ip: usize) -> Option<Symbol<A>> {
        debug_assert!(ip < self.code.len());

        match self.functions.locate(ip) {
            Function::Bytecode { mfa, offset, .. } => {
                let fp = *offset;
                let loc = self.debug_info.offset_to_source_location(fp, ip);
                Some(Symbol::Erlang { mfa: *mfa, loc })
            }
            Function::Bif { .. } | Function::Native { .. } => None,
        }
    }

    /// Attempts to symbolicate the given [`FunId`] in this bytecode module.
    ///
    /// NOTE: This function will panic if `id` is not valid in this module.
    #[inline]
    pub fn function_symbol(&self, id: FunId) -> Symbol<A> {
        match self.functions.get(id) {
            Function::Bytecode { mfa, offset, .. } => {
                let loc = self.debug_info.function_pointer_to_source_location(*offset);
                Symbol::Erlang { mfa: *mfa, loc }
            }
            Function::Bif { mfa, .. } => Symbol::Bif(*mfa),
            Function::Native { name, .. } => Symbol::Native(name.clone()),
        }
    }

    pub fn set_function_frame_size(&mut self, id: FunId, size: usize) {
        if let Function::Bytecode {
            ref mut frame_size, ..
        } = self.functions.get_mut(id)
        {
            *frame_size = size;
        } else {
            panic!("cannot set frame size of non-bytecoded function");
        }
    }

    /// Set the source location for the given [`FunId`]
    pub fn set_function_location(&mut self, id: FunId, loc: Location) -> Option<LocationId> {
        if let Function::Bytecode { offset, .. } = self.functions.get(id) {
            let id = self.debug_info.get_or_insert_location(loc);
            self.debug_info.register_offset(*offset, id);
            Some(id)
        } else {
            None
        }
    }

    /// Set the source location for a given instruction offset
    ///
    /// NOTE: This expects a `LocationId`, not a `Location`, this is because the expectation
    /// is that many instructions will share the same `Location`, and that should be cached by
    /// the caller by first inserting a `Location` to get a `LocationId`, then calling this function
    /// with that
    pub fn set_instruction_location(&mut self, ip: usize, loc: LocationId) {
        self.debug_info.register_offset(ip, loc);
    }

    /// Insert a source filename in the debug info, returning a `FileId` for use with `Location`
    #[inline]
    pub fn get_or_insert_file(&mut self, file: &str) -> FileId {
        self.debug_info.get_or_insert_file(file)
    }

    /// Insert a source location in the debug info, returning a `LocationId` for use with
    /// `set_instruction_location`
    #[inline]
    pub fn get_or_insert_location(&mut self, loc: Location) -> LocationId {
        self.debug_info.get_or_insert_location(loc)
    }
}
impl<A: Atom, T: AtomTable<Atom = A>> Eq for ByteCode<A, T> {}
impl<A: Atom, T: AtomTable<Atom = A>> PartialEq for ByteCode<A, T> {
    fn eq(&self, other: &Self) -> bool {
        if self.functions.ne(&other.functions) {
            return false;
        }
        if self.code.len() != other.code.len() {
            return false;
        }
        if self.code.iter().ne(other.code.iter()) {
            return false;
        }
        true
    }
}

/// This is a simple implementation of [`AtomTable`] which can be used for cases in which we don't
/// require a global atom table, or we don't have access to the runtime representation of atoms.
#[derive(Default)]
pub struct LocalAtomTable {
    ids: BTreeMap<&'static str, AtomicStr>,
    arena: DroplessArena,
}
impl fmt::Debug for LocalAtomTable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LocalAtomTable")
            .field("ids", &self.ids)
            .finish()
    }
}
impl PartialEq for LocalAtomTable {
    fn eq(&self, other: &Self) -> bool {
        self.ids.eq(&other.ids)
    }
}
impl AtomTable for LocalAtomTable {
    type Atom = AtomicStr;
    type AtomError = ();
    type Guard = Self;

    #[inline]
    fn len(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    fn iter<'a>(&'a self) -> impl Iterator<Item = Self::Atom> + 'a {
        self.ids.values().copied()
    }

    fn get_or_insert(&mut self, name: &str) -> Result<Self::Atom, Self::AtomError> {
        match self.ids.get(name).copied() {
            Some(existing) => Ok(existing),
            None => unsafe { self.insert(name) },
        }
    }

    #[inline]
    fn change<F, T>(&mut self, callback: F) -> T
    where
        F: FnOnce(&mut Self::Guard) -> T,
    {
        callback(self)
    }
}
impl LocalAtomTable {
    // This function is used to insert new atoms in the table during runtime
    // SAFETY: `name` must have been checked as not existing while holding the current mutable
    // reference.
    unsafe fn insert(&mut self, name: &str) -> Result<AtomicStr, ()> {
        use core::intrinsics::unlikely;

        if unlikely(name.len() == 0) {
            let data = AtomicStr {
                ptr: ptr::null(),
                size: 0,
            };
            self.ids.insert("", data);

            return Ok(data);
        }

        // Allocate memory for value
        let bytes = name.as_bytes();
        let layout = Layout::for_value(bytes);
        let ptr = self.arena.alloc_raw(layout.pad_to_align());

        let offset = ptr.align_offset(mem::align_of_val(bytes));
        let ptr = ptr.add(offset);
        // Write atom data
        ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());

        let str_ptr: *const str = ptr::from_raw_parts(ptr as *const (), name.len());

        // Register in atom table
        let string = unsafe { &*str_ptr };
        let data = AtomicStr {
            ptr,
            size: bytes.len(),
        };
        self.ids.insert(string, data);

        Ok(data)
    }
}

/// This is similar to [`LocalAtomTable`], but is designed for storing binaries/bitstrings.
///
/// This table is designed to be able to leak the arena, so that any references into the
/// table's data can outlive the bytecode container it was originally contained in.
#[derive(Default)]
pub struct BinaryTable {
    ids: BTreeMap<&'static [u8], NonNull<BinaryData>>,
    arena: DroplessArena,
}
impl fmt::Debug for BinaryTable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BinaryTable")
            .field("ids", &self.ids)
            .finish()
    }
}
impl PartialEq for BinaryTable {
    fn eq(&self, other: &Self) -> bool {
        self.ids.eq(&other.ids)
    }
}
impl BinaryTable {
    pub fn append(&mut self, other: &Self) {
        for (id, value) in other.ids.iter() {
            if self.ids.contains_key(id) {
                continue;
            }
            let flags = unsafe { value.as_ref().flags() };
            unsafe {
                self.insert(flags, *id);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = NonNull<BinaryData>> + '_ {
        self.ids.values().copied()
    }

    pub fn get_data_or_insert(&mut self, flags: BinaryFlags, bytes: &[u8]) -> NonNull<BinaryData> {
        match self.get_data(bytes) {
            Some(existing_id) => existing_id,
            None => unsafe { self.insert(flags, bytes) },
        }
    }

    fn get_data(&self, bytes: &[u8]) -> Option<NonNull<BinaryData>> {
        self.ids.get(bytes).copied()
    }

    // This function is used to insert new atoms in the table during runtime
    // SAFETY: `name` must have been checked as not existing while holding the current mutable
    // reference.
    unsafe fn insert(&mut self, flags: BinaryFlags, bytes: &[u8]) -> NonNull<BinaryData> {
        let empty = ptr::from_raw_parts::<BinaryData>(ptr::null() as *const (), bytes.len());
        let layout = unsafe { Layout::for_value_raw(empty) };
        let layout = layout.pad_to_align();

        let ptr = self.arena.alloc_raw(layout);
        let offset = ptr.align_offset(mem::align_of_val_raw(empty));
        let ptr = ptr.add(offset) as *mut ();
        let mut ptr: NonNull<BinaryData> =
            NonNull::new_unchecked(ptr::from_raw_parts_mut(ptr, bytes.len()));

        {
            let binary = ptr.as_mut();
            binary.set_flags(flags);
            binary.copy_from_slice(bytes);
            self.ids.insert(binary.as_bytes(), ptr);
        }

        ptr
    }
}

/// This table stores metadata about all the functions in its containing bytecode module
pub struct FunctionTable<A: Atom> {
    /// This contains a single unique instance of each functions' metadata, the index into
    /// this vector is what constitutes a [`FunId`]
    registered: Vec<Function<A>>,
    /// This maps all Erlang functions to their corresponding [`FunId`]
    id_by_mfa: BTreeMap<ModuleFunctionArity<A>, FunId>,
    /// This maps all native functions to their corresponding [`FunId`]
    id_by_name: BTreeMap<A, FunId>,
    /// This is a reverse lookup from instruction offset to [`FunId`]
    ///
    /// This is primarily intended for use in producing stacktraces from a series
    /// of instruction offsets representing the frames of each call in the stack.
    ///
    /// Instruction offsets which fall in the range between two function entries are
    /// considered to belong to the first function in that range.
    id_by_offset: BTreeMap<usize, FunId>,
}
impl<A: Atom + fmt::Debug> fmt::Debug for FunctionTable<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FunctionTable")
            .field("registered", &self.registered.as_slice())
            .finish()
    }
}
impl<A: Atom> Eq for FunctionTable<A> {}
impl<A: Atom> PartialEq<FunctionTable<A>> for FunctionTable<A> {
    fn eq(&self, other: &Self) -> bool {
        self.registered == other.registered
    }
}
impl<A: Atom> Default for FunctionTable<A> {
    fn default() -> Self {
        Self {
            registered: Vec::new(),
            id_by_mfa: BTreeMap::new(),
            id_by_name: BTreeMap::new(),
            id_by_offset: BTreeMap::new(),
        }
    }
}
impl<A: Atom> FunctionTable<A> {
    /// Returns the number of functions contained in this table
    pub fn len(&self) -> usize {
        self.registered.len()
    }

    /// Gets function metadata by [`FunId`]
    #[inline(always)]
    pub fn get(&self, id: FunId) -> &Function<A> {
        &self.registered[id as usize]
    }

    /// Gets function metadata mutably by [`FunId`]
    #[inline(always)]
    pub fn get_mut(&mut self, id: FunId) -> &mut Function<A> {
        &mut self.registered[id as usize]
    }

    /// Returns an iterator over the functions defined in this table
    pub fn iter<'a, 'b: 'a>(&'b self) -> impl Iterator<Item = &'b Function<A>> + 'a {
        self.registered.iter()
    }

    /// Get the instruction offset for the entry of the given function
    ///
    /// # SAFETY
    ///
    /// This function is intended for use by the emulator when looking up the instruction
    /// offset for a callee known to be a bytecoded function. It is undefined behavior to
    /// call this function with a [`FunId`] which points to a natively-implemented function.
    #[inline]
    pub unsafe fn offset(&self, id: FunId) -> usize {
        self.registered[id as usize].offset().unwrap_unchecked()
    }

    /// Get a function by [`ModuleFunctionArity`]
    ///
    /// This only works for Erlang functions (built-in or bytecoded)
    #[inline]
    pub fn find_by_mfa(&self, mfa: &ModuleFunctionArity<A>) -> Option<&Function<A>> {
        self.id_by_mfa.get(&mfa).copied().map(|id| self.get(id))
    }

    /// Get a function by symbol name as a string
    ///
    /// Only native functions are registered this way currently, use `get_by_mfa` for Erlang
    /// functions
    #[inline]
    pub fn find_by_name(&self, name: A) -> Option<&Function<A>> {
        self.id_by_name.get(&name).copied().map(|id| self.get(id))
    }

    /// Maps an instruction pointer/offset to the corresponding function to which it belongs
    #[inline]
    pub fn locate(&self, ip: usize) -> &Function<A> {
        self.id_by_offset
            .range(..=ip)
            .next_back()
            .map(|(_, id)| self.get(*id))
            .expect("invalid instruction pointer")
    }

    pub fn load(&mut self, function: Function<A>) {
        assert_eq!(self.registered.len(), function.id() as usize);

        match function {
            Function::Bytecode {
                id, mfa, offset, ..
            } => {
                self.id_by_mfa.insert(mfa, id);
                self.id_by_offset.insert(offset, id);
            }
            Function::Bif { id, mfa } => {
                self.id_by_mfa.insert(mfa, id);
            }
            Function::Native { id, name, .. } => {
                self.id_by_name.insert(name, id);
            }
        }

        self.registered.push(function);
    }

    pub fn define(
        &mut self,
        mfa: ModuleFunctionArity<A>,
        offset: usize,
    ) -> Result<FunId, InvalidBytecodeError<A>> {
        use alloc::collections::btree_map::Entry;

        match self.id_by_mfa.entry(mfa) {
            Entry::Vacant(entry) => {
                let id = self.registered.len() as FunId;
                self.registered.push(Function::Bytecode {
                    id,
                    is_nif: false,
                    mfa,
                    offset,
                    frame_size: 0,
                });
                entry.insert(id);
                if offset > 0 {
                    assert_eq!(self.id_by_offset.insert(offset, id), None);
                }
                Ok(id)
            }
            Entry::Occupied(entry) => {
                let id = *entry.get();
                let fun = unsafe { self.registered.get_unchecked_mut(id as usize) };
                match fun {
                    Function::Bytecode {
                        offset: ref mut offs,
                        ..
                    } if *offs == 0 => {
                        if offset > 0 {
                            *offs = offset;
                            assert_eq!(self.id_by_offset.insert(offset, id), None);
                        }
                        Ok(id)
                    }
                    Function::Bytecode { .. } if offset == 0 => Ok(id),
                    Function::Bytecode { .. } => {
                        Err(InvalidBytecodeError::DuplicateDefinition(mfa))
                    }
                    Function::Native { .. } => Err(InvalidBytecodeError::DuplicateDefinition(mfa)),
                    // When a BIF definition occurs, we replace the function type with the Bytecode
                    // type, but always set the `is_nif` flag to true, as in the usual case, the BIF
                    // definition in the module is a wrapper for the natively-implemented BIF
                    // function.
                    //
                    // At runtime, if the native symbol doesn't exist, this will dispatch to the
                    // bytecoded function
                    Function::Bif { mfa, .. } => {
                        let mfa = *mfa;
                        *fun = Function::Bytecode {
                            id,
                            mfa,
                            offset,
                            is_nif: true,
                            frame_size: 0,
                        };
                        Ok(id)
                    }
                }
            }
        }
    }

    pub fn get_or_define(&mut self, mfa: ModuleFunctionArity<A>) -> FunId {
        self.define(mfa, 0).unwrap()
    }

    pub fn get_or_define_bif(&mut self, mfa: ModuleFunctionArity<A>) -> FunId {
        use alloc::collections::btree_map::Entry;
        match self.id_by_mfa.entry(mfa) {
            Entry::Vacant(entry) => {
                let id = self.registered.len() as FunId;
                self.registered.push(Function::Bif { id, mfa });
                entry.insert(id);
                id
            }
            Entry::Occupied(entry) => *entry.get(),
        }
    }

    pub fn get_or_define_nif(&mut self, name: A, arity: u8) -> FunId {
        use alloc::collections::btree_map::Entry;
        let name = name.into();
        match self.id_by_name.entry(name) {
            Entry::Vacant(entry) => {
                let id = self.registered.len() as FunId;
                self.registered.push(Function::Native { id, name, arity });
                entry.insert(id);
                id
            }
            Entry::Occupied(entry) => *entry.get(),
        }
    }

    pub fn set_offset(&mut self, id: FunId, offset: usize) {
        match self.get_mut(id) {
            Function::Bytecode {
                offset: ref mut offs,
                ..
            } => {
                *offs = offset;
            }
            Function::Bif { .. } | Function::Native { .. } => panic!("invalid bytecode function id {} - offsets are not supported on natively-implemented functions", id),
        }
        self.id_by_offset.insert(offset, id);
    }

    pub fn append(
        &mut self,
        other: &Self,
        base_offset: usize,
    ) -> Result<(), InvalidBytecodeError<A>> {
        for fun in other.registered.iter() {
            match fun {
                Function::Native { name, arity, .. } => {
                    if !self.id_by_name.contains_key(name) {
                        let id = self.registered.len() as FunId;
                        self.registered.push(Function::Native {
                            id,
                            name: *name,
                            arity: *arity,
                        });
                        self.id_by_name.insert(*name, id);
                    }
                }
                Function::Bif { mfa, .. } => {
                    if !self.id_by_mfa.contains_key(mfa) {
                        let id = self.registered.len() as FunId;
                        self.registered.push(Function::Bif { id, mfa: *mfa });
                        self.id_by_mfa.insert(*mfa, id);
                    }
                }
                Function::Bytecode {
                    is_nif,
                    mfa,
                    offset,
                    frame_size,
                    ..
                } => {
                    let offset = *offset;
                    if !self.id_by_mfa.contains_key(mfa) {
                        let new_offset = if offset > 0 { offset + base_offset } else { 0 };
                        let id = self.registered.len() as FunId;
                        self.registered.push(Function::Bytecode {
                            id,
                            is_nif: *is_nif,
                            mfa: *mfa,
                            offset: new_offset,
                            frame_size: *frame_size,
                        });
                        self.id_by_mfa.insert(*mfa, id);
                        if offset > 0 {
                            self.id_by_offset.insert(new_offset, id);
                        }
                    } else {
                        let id = self.id_by_mfa[mfa];
                        let self_offset = self.registered[id as usize].offset().unwrap();
                        match (self_offset, offset) {
                            // Either the current definition is canonical, or neither are, so ignore
                            (_, 0) => continue,
                            // The merging definition is canonical, so apply the changes
                            (0, offset) => {
                                self.id_by_offset.insert(offset, id);
                                self.registered[id as usize] = Function::Bytecode {
                                    id,
                                    is_nif: *is_nif,
                                    mfa: *mfa,
                                    frame_size: *frame_size,
                                    offset: base_offset + offset,
                                };
                            }
                            // They conflict, raise an error
                            (_, _) => return Err(InvalidBytecodeError::DuplicateDefinition(*mfa)),
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
