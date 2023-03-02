use alloc::vec::Vec;
use core::mem;
use core::num::NonZeroU32;

use firefly_binary::{BinaryEntrySpecifier, Endianness};
use firefly_number::BigInt;

use crate::reader::{eof_to_invalid, Decode};
#[cfg(any(test, feature = "std"))]
use crate::writer::Encode;

use super::*;

/// For information about a specific opcode, see the documentation on its associated struct.
#[derive(Debug, Clone, PartialEq)]
#[repr(u8)]
pub enum Opcode<A: Atom> {
    Nop(Nop) = 0,
    Mov(Mov),
    Cmov(Cmov),
    Ret(Ret),
    Br(Br),
    Brz(Brz),
    Brnz(Brnz),
    JumpTable(JumpTable),
    JumpTableEntry(JumpTableEntry),
    Call(Call),
    CallApply2(CallApply2),
    CallApply3(CallApply3),
    CallNative(CallNative),
    CallStatic(CallStatic),
    CallIndirect(CallIndirect),
    Enter(Enter),
    EnterApply2(EnterApply2),
    EnterApply3(EnterApply3),
    EnterNative(EnterNative),
    EnterStatic(EnterStatic),
    EnterIndirect(EnterIndirect),
    IsAtom(IsAtom),
    IsBool(IsBool),
    IsNil(IsNil),
    IsTuple(IsTuple),
    IsTupleFetchArity(IsTupleFetchArity),
    IsMap(IsMap),
    IsCons(IsCons),
    IsList(IsList),
    IsInt(IsInt),
    IsFloat(IsFloat),
    IsNumber(IsNumber),
    IsPid(IsPid),
    IsRef(IsRef),
    IsPort(IsPort),
    IsBinary(IsBinary),
    IsFunction(IsFunction),
    LoadNil(LoadNil),
    LoadBool(LoadBool),
    LoadAtom(LoadAtom<A>),
    LoadInt(LoadInt),
    LoadBig(LoadBig),
    LoadFloat(LoadFloat),
    LoadBinary(LoadBinary),
    LoadBitstring(LoadBitstring),
    Not(Not),
    And(And),
    AndAlso(AndAlso),
    Or(Or),
    OrElse(OrElse),
    Xor(Xor),
    Bnot(Bnot),
    Band(Band),
    Bor(Bor),
    Bxor(Bxor),
    Bsl(Bsl),
    Bsr(Bsr),
    Div(Div),
    Rem(Rem),
    Neg(Neg),
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Divide(Divide),
    ListAppend(ListAppend),
    ListRemove(ListRemove),
    Eq(IsEq),
    Neq(IsNeq),
    Gt(IsGt),
    Gte(IsGte),
    Lt(IsLt),
    Lte(IsLte),
    Cons(Cons),
    Split(Split),
    Head(Head),
    Tail(Tail),
    Closure(Closure),
    UnpackEnv(UnpackEnv),
    Tuple(Tuple),
    TupleWithCapacity(TupleWithCapacity),
    TupleArity(TupleArity),
    GetElement(GetElement),
    SetElement(SetElement),
    SetElementMut(SetElementMut),
    Map(Map),
    MapPut(MapPut),
    MapPutMut(MapPutMut),
    MapUpdate(MapUpdate),
    MapUpdateMut(MapUpdateMut),
    MapExtendPut(MapExtendPut),
    MapExtendUpdate(MapExtendUpdate),
    MapTryGet(MapTryGet),
    Catch(Catch),
    EndCatch(EndCatch),
    LandingPad(LandingPad),
    StackTrace(StackTrace),
    Raise(Raise),
    Send(SendOp),
    RecvPeek(RecvPeek),
    RecvNext(RecvNext),
    RecvWait(RecvWait),
    RecvTimeout(RecvTimeout),
    RecvPop(RecvPop),
    Await(Await),
    Yield(Yield),
    GarbageCollect(GarbageCollect),
    NormalExit(NormalExit),
    ContinueExit(ContinueExit),
    Exit1(Exit1),
    Exit2(Exit2),
    Error1(Error1),
    Throw1(Throw1),
    Halt(Halt),
    BsInit(BsInit),
    BsPush(BsPush),
    BsFinish(BsFinish),
    BsMatchStart(BsMatchStart),
    BsMatch(BsMatch),
    BsMatchSkip(BsMatchSkip),
    BsTestTail(BsTestTail),
    FuncInfo(FuncInfo),
    Identity(Identity),
    Spawn2(Spawn2),
    Spawn3(Spawn3),
    Spawn3Indirect(Spawn3Indirect),
    Trap(Trap),
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for bool {
    #[inline]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        Ok((reader.read_byte().map_err(eof_to_invalid)? > 0) as bool)
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for bool {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_byte((*self) as u8)
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for u8 {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader.read_byte().map_err(eof_to_invalid)
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for u8 {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_byte(*self)
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for u16 {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader.read_integer().map_err(eof_to_invalid)
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for u16 {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_integer(*self)
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for i16 {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader.read_integer().map_err(eof_to_invalid)
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for i16 {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_integer(*self)
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for u32 {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader.read_integer().map_err(eof_to_invalid)
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for u32 {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_integer(*self)
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for usize {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader.read_integer().map_err(eof_to_invalid)
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for usize {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_integer(*self)
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for i64 {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader.read_integer().map_err(eof_to_invalid)
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for i64 {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_integer(*self)
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for f64 {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader.read_float().map_err(eof_to_invalid)
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for f64 {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_float(*self)
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for Option<NonZeroU32> {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_integer(self.map(|n| n.get()).unwrap_or(0))
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for Option<NonZeroU32> {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader
            .read_integer()
            .map(NonZeroU32::new)
            .map_err(eof_to_invalid)
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for Option<Register> {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        assert_ne!(*self, Some(Register::MAX));
        writer.write_integer((*self).unwrap_or(Register::MAX))
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for Option<Register> {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader
            .read_integer()
            .map(|n| if n == Register::MAX { None } else { Some(n) })
            .map_err(eof_to_invalid)
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for SpawnOpts {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_integer(self.bits())
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for SpawnOpts {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader
            .read_integer()
            .map_err(eof_to_invalid)
            .and_then(|n| SpawnOpts::from_bits(n).ok_or(ReadError::Invalid))
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for Endianness {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_integer((*self) as u8)
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for Endianness {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader
            .read_integer()
            .map_err(eof_to_invalid)
            .and_then(|n: u8| n.try_into().map_err(|_| ReadError::Invalid))
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for BinaryEntrySpecifier {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        match self {
            Self::Integer {
                signed,
                endianness,
                unit,
            } => {
                writer.write_integer(0u8)?;
                writer.write_integer(*signed as u8)?;
                writer.write_integer(*endianness as u8)?;
                writer.write_integer(*unit)
            }
            Self::Float { endianness, unit } => {
                writer.write_integer(1u8)?;
                writer.write_integer(*endianness as u8)?;
                writer.write_integer(*unit)
            }
            Self::Binary { unit } => {
                writer.write_integer(2u8)?;
                writer.write_integer(*unit)
            }
            Self::Utf8 => writer.write_integer(3u8),
            Self::Utf16 { endianness } => {
                writer.write_integer(4u8)?;
                writer.write_integer(*endianness as u8)
            }
            Self::Utf32 { endianness } => {
                writer.write_integer(5u8)?;
                writer.write_integer(*endianness as u8)
            }
        }
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for BinaryEntrySpecifier {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        let tag = reader.read_byte().map_err(eof_to_invalid)?;
        match tag {
            0 => {
                let signed = reader.read_byte().map_err(eof_to_invalid)? != 0;
                let endianness = reader.read_byte().map_err(eof_to_invalid)?.try_into()?;
                let unit = reader.read_byte().map_err(eof_to_invalid)?;
                Ok(Self::Integer {
                    signed,
                    endianness,
                    unit,
                })
            }
            1 => {
                let endianness = reader.read_byte().map_err(eof_to_invalid)?.try_into()?;
                let unit = reader.read_byte().map_err(eof_to_invalid)?;
                Ok(Self::Float { endianness, unit })
            }
            2 => {
                let unit = reader.read_byte().map_err(eof_to_invalid)?;
                Ok(Self::Binary { unit })
            }
            3 => Ok(Self::Utf8),
            4 => {
                let endianness = reader.read_byte().map_err(eof_to_invalid)?.try_into()?;
                Ok(Self::Utf16 { endianness })
            }
            5 => {
                let endianness = reader.read_byte().map_err(eof_to_invalid)?.try_into()?;
                Ok(Self::Utf16 { endianness })
            }
            _ => Err(ReadError::Invalid),
        }
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for BsMatchSkipType {
    #[inline]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader
            .read_byte()
            .map_err(eof_to_invalid)
            .and_then(|b| b.try_into().map_err(|_| ReadError::Invalid))
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for BsMatchSkipType {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_byte(*self as u8)
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for ErrorKind {
    #[inline]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        let tag = reader.read_byte().map_err(eof_to_invalid)?;
        Ok(unsafe { mem::transmute::<_, ErrorKind>(tag) })
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for ErrorKind {
    #[inline(always)]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_byte(*self as u8)
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for A {
    #[inline]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        let offset = reader.read_integer().map_err(eof_to_invalid)?;
        Ok(reader.atom_from_offset(offset))
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for A {
    #[inline]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        let (ptr, _) = A::into_raw_parts(self.unpack());
        writer.write_integer(ptr as usize)
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for BigInt {
    #[inline]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        let size = reader.read_integer().map_err(eof_to_invalid)?;
        let bytes = reader.read_bytes(size).map_err(eof_to_invalid)?;
        Ok(BigInt::from_signed_bytes_be(bytes))
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for BigInt {
    #[inline]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        let bytes = self.to_signed_bytes_be();
        writer.write_integer(bytes.len())?;
        writer.write_all(bytes.as_slice())
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for *const BinaryData {
    #[inline]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        let offset: usize = reader.read_integer().map_err(eof_to_invalid)?;
        Ok(reader.binary_from_offset(offset))
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for *const BinaryData {
    #[inline]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        let (ptr, _) = (*self).to_raw_parts();
        writer.write_integer(ptr as usize)
    }
}

impl<P, A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for *const P {
    #[inline]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        let offset: usize = reader.read_integer().map_err(eof_to_invalid)?;
        Ok(offset as *const _)
    }
}

#[cfg(any(test, feature = "std"))]
impl<P, A: Atom> Encode<A> for *const P {
    #[inline]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        let (ptr, _meta) = (*self).to_raw_parts();
        writer.write_integer(ptr as usize)
    }
}

impl<U, A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for Vec<U>
where
    U: Decode<A, T>,
{
    #[inline]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        let len: usize = reader.read_integer().map_err(eof_to_invalid)?;
        let mut items = Vec::with_capacity(len);
        for _ in 0..len {
            items.push(U::decode(reader)?);
        }
        Ok(items)
    }
}

#[cfg(any(test, feature = "std"))]
impl<T, A: Atom> Encode<A> for Vec<T>
where
    T: Encode<A>,
{
    #[inline]
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        let len = self.len();
        writer.write_integer(len)?;
        for elem in self.iter() {
            elem.encode(writer)?;
        }
        Ok(())
    }
}

macro_rules! encode_impl {
    ($ty:ty) => {
        #[cfg(any(test, feature = "std"))]
        impl<A: Atom> Encode<A> for $ty {
            #[inline(always)]
            fn encode(&self, _writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
                Ok(())
            }
        }
    };

    ($ty:ty, $($reg:ident),+) => {
        #[cfg(any(test, feature = "std"))]
        impl<A: Atom> Encode<A> for $ty {
            #[inline]
            fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
                $(
                    self.$reg.encode(writer)?;
                )*
                Ok(())
            }
        }
    };
}

macro_rules! decode_impl {
    ($ty:ty) => {
        impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for $ty {
            #[inline(always)]
            fn decode(_reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
                Ok(Self)
            }
        }
    };

    ($ty:ty, $($reg:ident),+) => {
        impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for $ty {
            #[inline(always)]
            fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
                $(
                    let $reg = Decode::decode(reader)?;
                )*
                Ok(Self {
                    $($reg),*
                })
            }
        }
    };
}

macro_rules! encoding_impl {
    ($ty:ty) => {
        encode_impl!($ty);
        decode_impl!($ty);
    };

    ($ty:ty, $($reg:ident),+) => {
        encode_impl!($ty, $($reg),*);
        decode_impl!($ty, $($reg),*);
    };
}

/// A no-op instruction
///
/// This instruction has no effect whatsoever, the program will immediately
/// continue to the next instruction following this one.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Nop;
encoding_impl!(Nop);

/// Returns from the current call frame, placing the value in `reg` into the return register.
///
/// This instruction can be thought of as performing the following sequence of operations:
///
/// * Moves `reg` to the return register in the current frame, which corresponds to the register
/// in which the caller expects the return value to be placed.
/// * Gets the continuation pointer/return address stored in the current frame on entry
/// * Restores the frame and stack pointers of the caller, which were stored in the current frame on
///   entry
/// * Jumps to the continuation pointer
///
/// However there is also some implicit behaviour here at the top of the stack, i.e. when returning
/// from the initial function that a process was spawned with. ByteCode modules always have the
/// following instructions at the very start of the code region:
///
/// ```text,ignore
/// 0 | Nop
/// 1 | NormalExit
/// 2 | ContinueExit
/// . | ..
/// . | ..
/// ```
///
/// When a process is spawned, the continuation pointer of the first frame is set to zero, which
/// means that when we return from the initial function, we will jump to the `Nop`, which will then
/// fall through to `NormalExit`, which corresponds to the behavior we want when a process returns
/// from its initial function normally.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Ret {
    pub reg: Register,
}
encoding_impl!(Ret, reg);

/// Jumps unconditionally to the instruction at `offset` from the current instruction pointer
///
/// NOTE: This instruction is sensitive to any changes to the code section, i.e. if linking two
/// bytecode modules together, instructions such as `Br` in one of them must have their offsets
/// adjusted to account for their new location in the merged module.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Br {
    pub offset: JumpOffset,
}
encoding_impl!(Br, offset);

/// If the value in `reg` is false, jumps to the instruction at `offset` from the current
/// instruction pointer
///
/// NOTE: This instruction is sensitive to any changes to the code section, i.e. if linking two
/// bytecode modules together, instructions such as `Brz` in one of them must have their offsets
/// adjusted to account for their new location in the merged module.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Brz {
    pub reg: Register,
    pub offset: JumpOffset,
}
encoding_impl!(Brz, reg, offset);

/// If the value in `reg` is true, jumps to the instruction at `offset` from the current instruction
/// pointer
///
/// NOTE: This instruction is sensitive to any changes to the code section, i.e. if linking two
/// bytecode modules together, instructions such as `Brnz` in one of them must have their offsets
/// adjusted to account for their new location in the merged module.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Brnz {
    pub reg: Register,
    pub offset: JumpOffset,
}
encoding_impl!(Brnz, reg, offset);

/// Performs a multi-way branch over `len` branches, using `reg` to select an appropriate branch.
/// If none of the branches match, then control transfers to the instruction following the last
/// entry in the jump table. The value in `reg` must be a 32-bit integer value, as the jump table
/// entries are all 32-bit integer immediates.
///
/// This instruction is used to implement `switch`, but can be used for any jump table conditional
/// on some source value.
///
/// This instruction is immediately followed by `len` `JumpTableEntry` instructions, which are
/// semantically equivalent to `Br`, but carry a 32-bit integer immediate value used by `JumpTable`
/// to determine whether or not to dispatch to that specific instruction or not.
///
/// NOTE: Our emulator never dispatches `JumpTableEntry`, instead the implementation of `JumpTable`
/// handles selecting a branch and jumping directly.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct JumpTable {
    pub reg: Register,
    pub len: u8,
}
encoding_impl!(JumpTable, reg, len);

/// Represents a single entry in a `JumpTable`.
///
/// This instruction is equivalent to a `Br` when dispatched, but is its own instruction because it
/// also carries the sentinel value used by its parent jump table. While it is not invalid per-se to
/// use this instruction without a `JumpTable`, it makes no sense to do so when you can use `Br`
/// instead.
///
/// NOTE: This instruction is sensitive to any changes to the code section, i.e. if linking two
/// bytecode modules together, instructions such as `JumpTableEntry` in one of them must have their
/// offsets adjusted to account for their new location in the merged module.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct JumpTableEntry {
    pub imm: u32,
    pub offset: JumpOffset,
}
encoding_impl!(JumpTableEntry, imm, offset);

/// Calls a bytecode-compiled function whose `FuncInfo` instruction is found at `offset`,
/// an absolute offset from the start of the code section.
///
/// # Stack Layout
///
/// To understand this instruction, and other call instructions, let's take a look at the way our
/// stack is expected to be laid out. In the following, the stack grows downwards to make things a
/// bit easier to read. Let's assume that we're looking at the stack for the following Erlang code:
///
/// ```erlang
/// main() ->
///   Result = fib(3),
///   erlang:display(Result).
///
/// fib(0) -> 0;
/// fib(1) -> 1;
/// fib(N) -> fib(N - 1) + fib(N - 2).
/// ```
///
/// The corresponding bytecode might look something like this:
///
/// ```text
/// 0  | nop
/// 1  | normal_exit
/// 2  | continue_exit
/// 3  | func_info              # fn main/0
/// 4  | load_int $4, 3
/// 5  | call $2, fib/1
/// 6  | enter erlang:display/1
/// 7  | func_info              # fn fib/1
/// 8  | load_int $3, 0
/// 9  | is_eq $4, $2, $3
/// 10 | brnz $4, 24
/// 11 | load_int $5, 1
/// 12 | is_eq $6, $2, $5
/// 13 | brnz $6, 26
/// 14 | load_int $7, 1
/// 15 | sub $8, $2, $7
/// 16 | mov $11, $8
/// 17 | call $9, fib/1
/// 18 | load_int $12, 2
/// 19 | sub $13, $2, $9
/// 20 | mov $16, $13
/// 21 | call $14, fib/1
/// 22 | add $15, $9, $14
/// 23 | ret $15
/// 24 | load_int $0, 0
/// 25 | ret $0
/// 26 | load_int $0, 1
/// 27 | ret $1
/// 28 | func_info              # fn erlang:display/1
///    ..omitted..
/// ```
///
/// The stack will look like the following, just prior to returning from the first call to `fib(N -
/// 2)`.
///
/// ```text
/// main() -> 0  | 0   # return
///           1  | 0   # cp, instruction where control transfers when main/0 returns
/// fib(3) -> 2  | 0   # return
///           3  | 6   # cp, instruction where control transfers when fib/3 returns
///           4  | 3   # arg0
///           .  | ..  # stack space allocated for locals
///           .  | ..
/// fib(1) -> 14 | 1   # return
///           15 | 22  # cp, instruction where control transfers when `fib/(N - 2)` returns
///           16 | 1   # arg0
///           .  |
/// sp ->     29 |
/// ```
///
/// As you can see, the start of each call frame begins with the return register, which is placed
/// at the register where the return value is expected by the caller. This is then followed by
/// the continuation pointer, which is used to unwind the stack both for returns and exceptions.
/// Following that are all of the arguments to the callee, in order of appearance. All stack space
/// following the arguments is allocated by the `func_info` instruction on entry, and is used for
/// locals during execution of the call.
///
/// Because a call may occur during execution of a function, all stack space above the return
/// register must be treated by the caller as if it might be clobbered by the callee, as it almost
/// certainly will.
///
/// # Semantics
///
/// So now that we've looked at how our call stack works, the `Call` instruction is semantically
/// equivalent to the following sequence of operations:
///
/// * Save the callers frame pointer and current stack pointer somewhere (our runtime uses stack
///   marks stored adjacent to the stack itself)
/// * Adjust the current frame pointer to `dest`
/// * Set the current stack pointer equal to the frame pointer + 2 (to reserve the registers for the
///   return value and continuation pointer)
/// * Save the current instruction pointer to the continuation pointer register
/// * Jump to the `FuncInfo` instruction at `offset`
///
/// NOTE: It is expected that the bytecode is generated in such a way that the arguments are already
/// moved in to place relative to `dest` such that the first argument is in the register immediately
/// following the continuation pointer register. The `FuncInfo` instruction will adjust the stack
/// pointer based on the function arity and frame size.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Call {
    pub dest: Register,
    pub offset: usize,
}
encoding_impl!(Call, dest, offset);

/// This instruction implements `erlang:apply/2`.
///
/// This instruction is distinct from `CallIndirect`, as that instruction follows the same
/// calling convention as `Call`, whereas this must extract arguments from the provided
/// argument list, set up the stack properly, then delegate to `CallIndirect` for
/// the actual call itself. In other words, this instruction is a superinstruction which
/// is semantically equivalent to `CallIndirect`, but includes additional operations to
/// validate and extract arguments to the stack in preparation for `CallIndirect`.
///
/// Like all `Call`-like opcodes, the callee executes in a new stack frame.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CallApply2 {
    pub dest: Register,
    pub callee: Register,
    pub argv: Register,
}
encoding_impl!(CallApply2, dest, callee, argv);

/// This instruction implements `erlang:apply/3`.
///
/// This instruction is virtually identical to `CallApply2`, except rather than being a
/// superinstruction of `CallIndirect`, it is instead a superinstruction of `CallStatic`,
/// with some additional validation done at runtime because the callee cannot be resolved
/// statically.
///
/// Like all `Call`-like opcodes, the callee executes in a new stack frame.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CallApply3 {
    pub dest: Register,
    pub module: Register,
    pub function: Register,
    pub argv: Register,
}
encoding_impl!(CallApply3, dest, module, function, argv);

/// This instruction is used to facilitate calls to natively-implemented functions which are not
/// BIFs.
///
/// It expects that the stack is set up following the same calling convention as other `Call`-like
/// instructions, but the callee is executed using the native stack, not the process stack.
/// Arguments to the callee are passed using the native C calling convention.
///
/// This semantically executes in a new stack frame, but as mentioned above, does not use the
/// process stack during execution of the callee.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CallNative {
    pub dest: Register,
    pub callee: *const (),
    pub arity: Arity,
}
encoding_impl!(CallNative, dest, callee, arity);

/// This instruction is used to represent calls to a statically-resolved callee when we can't
/// commit to a more specialized call operation. For example, we might have bytecode functions
/// which may have native implementations, but we only know if the native implementation exists
/// at runtime, so we must defer resolution until runtime. Likewise, we may not know whether a
/// BIF is implemented via bytecode or native code, so that is deferred until runtime.
///
/// Semantically this is equivalent to `Call` or `CallNative`, depending on how the callee is
/// implemented at runtime. When a bytecode module is compiled, we attempt to convert
/// all `CallStatic` ops to `Call` or `CallNative` if possible. If unable to do so, then at runtime
/// the implementation looks up the function and dispatches to the appropriate specialized
/// instruction for the call.
///
/// As with all `Call`-like ops, this executes in a new stack frame.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CallStatic {
    pub dest: Register,
    pub callee: FunId,
}
encoding_impl!(CallStatic, dest, callee);

/// This instruction is like `Call`, except the callee is indirect, i.e. a function value.
/// With additional optimization passes, we could likely transform some `CallIndirect` instructions
/// to `Call`, but the common case is that indirect calls are to a closure, in which case the
/// calling convention is slightly different.
///
/// When calling a closure, the calling convention is identical to `Call`, except that the closure
/// itself is passed as an implicit extra argument. This closure value is then used to unpack the
/// closure environment on entry to the closure body. This extra argument is only passed for fat
/// closures however, _not_ thin closures, which have no environment, and as such have no need for
/// the extra argument.
///
/// The callee is not guaranteed to be a function, so the implementation of this instruction must
/// validate the callee value and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CallIndirect {
    pub dest: Register,
    pub callee: Register,
    pub arity: Arity,
}
encoding_impl!(CallIndirect, dest, callee, arity);

/// This instruction (and all other `Enter`-like instructions) is largely equivalent to `Call`, but
/// they differ in how they affect the stack. Specifically, `Enter` represents a call in tail
/// position, where the caller's stack frame is reused for the callee, rather than pushing a new
/// call frame on the stack.
///
/// As a result, `Enter` has no `dest` register, as it uses the return register of the current stack
/// frame. Similarly, `Enter` does not save a continuation pointer on the stack, instead, when the
/// callee returns it returns to the caller's continuation. To make things a bit clearer, consider
/// the following:
///
/// ```erlang
/// main() ->
///   Result = a(),
///   erlang:display(Result).
///
/// a() -> b().
///
/// b() -> "hi".
/// ```
///
/// During execution of this program, `b` is called in tail position by `a`, so when returning from
/// `b`, we do not return to `a`, but instead all the way back to `main`.
///
/// While more memory efficient, there can be some additional overhead when recursively calling the
/// current function and moves are needed to avoid clobbering registers holding arguments. This is a
/// small amount of overhead, but something to be aware of.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Enter {
    pub offset: usize,
}
encoding_impl!(Enter, offset);

/// Equivalent to `CallApply2`, but with the stack semantics of `Enter`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EnterApply2 {
    pub callee: Register,
    pub argv: Register,
}
encoding_impl!(EnterApply2, callee, argv);

/// Equivalent to `CallApply3`, but with the stack semantics of `Enter`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EnterApply3 {
    pub module: Register,
    pub function: Register,
    pub argv: Register,
}
encoding_impl!(EnterApply3, module, function, argv);

/// Equivalent to `CallNative`, but with the stack semantics of `Enter`
///
/// NOTE: Because native functions are called on the native stack, not the process stack,
/// it is not really accurate to say that we are reusing the caller's frame, however the
/// semantics are still useful, as the process stack is not manipulated by `EnterNative`
/// whereas `CallNative` still sets up a new call frame for stack traces.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EnterNative {
    pub callee: *const (),
    pub arity: Arity,
}
encoding_impl!(EnterNative, callee, arity);

/// Equivalent to `CallStatic`, but with the stack semantics of `Enter`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EnterStatic {
    pub callee: FunId,
}
encoding_impl!(EnterStatic, callee);

/// Equivalent to `CallIndirect`, but with the stack semantics of `Enter`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EnterIndirect {
    pub callee: Register,
    pub arity: Arity,
}
encoding_impl!(EnterIndirect, callee, arity);

/// Tests `value` to see if it holds an atom, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsAtom {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsAtom, dest, value);

/// Tests `value` to see if it holds a boolean, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsBool {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsBool, dest, value);

/// Tests `value` to see if it holds nil, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsNil {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsNil, dest, value);

/// Tests `value` to see if it holds a tuple, and puts the result in `dest`
///
/// If `arity` is provided, the test also semantically includes a test of the arity.
/// The overall result is only true if both tests are true. If `arity` is unset,
/// then any arity is considered a match.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsTuple {
    pub dest: Register,
    pub value: Register,
    pub arity: Option<NonZeroU32>,
}
encoding_impl!(IsTuple, dest, value, arity);

/// This is a fused instruction which tests whether `value` holds a tuple, placing the
/// result in `dest`; but additionaly, if the value is a tuple, loads the arity of that
/// tuple into `arity`.
///
/// This is commonly used in pattern matching, hence the benefit of fusing the `IsTuple`,
/// and `TupleArity` instructions.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsTupleFetchArity {
    pub dest: Register,
    pub arity: Register,
    pub value: Register,
}
encoding_impl!(IsTupleFetchArity, dest, arity, value);

/// Tests `value` to see if it holds a map, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsMap {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsMap, dest, value);

/// Tests `value` to see if it holds a non-empty list, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsCons {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsCons, dest, value);

/// Tests `value` to see if it holds a list, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsList {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsList, dest, value);

/// Tests `value` to see if it holds an integer, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsInt {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsInt, dest, value);

/// Tests `value` to see if it holds a float, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsFloat {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsFloat, dest, value);

/// Tests `value` to see if it holds any numeric value, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsNumber {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsNumber, dest, value);

/// Tests `value` to see if it holds a pid, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsPid {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsPid, dest, value);

/// Tests `value` to see if it holds a reference, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsRef {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsRef, dest, value);

/// Tests `value` to see if it holds a port, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsPort {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsPort, dest, value);

/// Tests `value` to see if it holds a binary/bitstring (depending on unit), and puts the result in
/// `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsBinary {
    pub dest: Register,
    pub value: Register,
    pub unit: u8,
}
encoding_impl!(IsBinary, dest, value, unit);

/// Tests `value` to see if it holds a function (optionally with `arity`), and puts the result in
/// `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsFunction {
    pub dest: Register,
    pub value: Register,
    pub arity: Option<Register>,
}
encoding_impl!(IsFunction, dest, value, arity);

/// Loads the nil value into `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LoadNil {
    pub dest: Register,
}
encoding_impl!(LoadNil, dest);

/// Loads a boolean value into `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LoadBool {
    pub dest: Register,
    pub value: bool,
}
encoding_impl!(LoadBool, dest, value);

/// Loads an atom value into `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LoadAtom<A: Atom> {
    pub dest: Register,
    pub value: A,
}
encoding_impl!(LoadAtom<A>, dest, value);

/// Loads a small integer value into `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LoadInt {
    pub dest: Register,
    pub value: i64,
}
encoding_impl!(LoadInt, dest, value);

/// Loads a big integer value into `dest`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadBig {
    pub dest: Register,
    pub value: BigInt,
}
encoding_impl!(LoadBig, dest, value);

/// Loads a floating point value into `dest`
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct LoadFloat {
    pub dest: Register,
    pub value: f64,
}
encoding_impl!(LoadFloat, dest, value);

/// Loads a binary value into `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LoadBinary {
    pub dest: Register,
    pub value: *const BinaryData,
}
encoding_impl!(LoadBinary, dest, value);

/// Loads a bitstring value into `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LoadBitstring {
    pub dest: Register,
    pub value: *const BinaryData,
}
encoding_impl!(LoadBitstring, dest, value);

/// Inverts the boolean value in `cond` and place it in `dest`
///
/// NOTE: It is not guaranteed that `cond` is a boolean, implementations must validate this and
/// raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Not {
    pub dest: Register,
    pub cond: Register,
}
encoding_impl!(Not, dest, cond);

/// Place the logical 'AND' of `lhs` and `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are boolean, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct And {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(And, dest, lhs, rhs);

/// Place the short-circuiting logical 'AND' of `lhs` and `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are boolean, implementations must validate one or
/// both arguments depending on whether `lhs` is true, and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AndAlso {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(AndAlso, dest, lhs, rhs);

/// Place the logical 'OR' of `lhs` and `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are boolean, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Or {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Or, dest, lhs, rhs);

/// Place the short-circuiting logical 'OR' of `lhs` and `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are boolean, implementations must validate one or
/// both arguments depending on whether `lhs` is false, and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct OrElse {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(OrElse, dest, lhs, rhs);

/// Place the logical 'XOR' of `lhs` and `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are boolean, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Xor {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Xor, dest, lhs, rhs);

/// Place the bitwise 'NOT' of `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are integers, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bnot {
    pub dest: Register,
    pub rhs: Register,
}
encoding_impl!(Bnot, dest, rhs);

/// Place the bitwise 'AND' of `lhs` and `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are integers, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Band {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Band, dest, lhs, rhs);

/// Place the bitwise 'OR' of `lhs` and `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are integers, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bor {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Bor, dest, lhs, rhs);

/// Place the bitwise 'XOR' of `lhs` and `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are integers, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bxor {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Bxor, dest, lhs, rhs);

/// Place the arithmetic bitshift-left of `value` by `shift` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are integers, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bsl {
    pub dest: Register,
    pub value: Register,
    pub shift: Register,
}
encoding_impl!(Bsl, dest, value, shift);

/// Place the bitshift-right of `value` by `shift` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are integers, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bsr {
    pub dest: Register,
    pub value: Register,
    pub shift: Register,
}
encoding_impl!(Bsr, dest, value, shift);

/// Place the result of dividing the integer `value` by `divisor` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are integers, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Div {
    pub dest: Register,
    pub value: Register,
    pub divisor: Register,
}
encoding_impl!(Div, dest, value, divisor);

/// Place the remainder of dividing the integer `value` by `divisor` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are integers, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Rem {
    pub dest: Register,
    pub value: Register,
    pub divisor: Register,
}
encoding_impl!(Rem, dest, value, divisor);

/// Place the result of negating `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the argument is a number, implementations must validate this and
/// raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Neg {
    pub dest: Register,
    pub rhs: Register,
}
encoding_impl!(Neg, dest, rhs);

/// Place the result of adding `lhs` and `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are numeric, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Add {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Add, dest, lhs, rhs);

/// Place the result of subtracting `lhs` and `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are numeric, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Sub {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Sub, dest, lhs, rhs);

/// Place the result of multiplying `lhs` and `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are numeric, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Mul {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Mul, dest, lhs, rhs);

/// Place the result of dividing `lhs` and `rhs` in `dest`
///
/// NOTE: It is not guaranteed that the arguments are numeric, implementations must validate this
/// and raise `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Divide {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Divide, dest, lhs, rhs);

/// Place the result of appending `rhs` to `list` in `dest`
///
/// NOTE: It is not guaranteed that `list` is a list, implementations must validate this and raise
/// `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ListAppend {
    pub dest: Register,
    pub list: Register,
    pub rhs: Register,
}
encoding_impl!(ListAppend, dest, list, rhs);

/// Place the result of removing `rhs` from `list` in `dest`
///
/// NOTE: It is not guaranteed that `list` is a list, implementations must validate this and raise
/// `badarg` if invalid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ListRemove {
    pub dest: Register,
    pub list: Register,
    pub rhs: Register,
}
encoding_impl!(ListRemove, dest, list, rhs);

/// Place the result of comparing `lhs` and `rhs` for equality in `dest`
///
/// If `strict` is true, this performs a strict equality comparison
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsEq {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
    pub strict: bool,
}
encoding_impl!(IsEq, dest, lhs, rhs, strict);

/// Place the result of comparing `lhs` and `rhs` for inequality in `dest`
///
/// If `strict` is true, this performs a strict inequality comparison
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsNeq {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
    pub strict: bool,
}
encoding_impl!(IsNeq, dest, lhs, rhs, strict);

/// Place the result of `lhs > rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsGt {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(IsGt, dest, lhs, rhs);

/// Place the result of `lhs >= rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsGte {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(IsGte, dest, lhs, rhs);

/// Place the result of `lhs < rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsLt {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(IsLt, dest, lhs, rhs);

/// Place the result of `lhs <= rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsLte {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(IsLte, dest, lhs, rhs);

/// Constructs a new cons cell from `head` and `tail`, placing it in `dest`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Cons {
    pub dest: Register,
    pub head: Register,
    pub tail: Register,
}
encoding_impl!(Cons, dest, head, tail);

/// Load the head of a cons cell into `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Head {
    pub dest: Register,
    pub list: Register,
}
encoding_impl!(Head, dest, list);

/// Load the tail of a cons cell into `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Tail {
    pub dest: Register,
    pub list: Register,
}
encoding_impl!(Tail, dest, list);

/// Load the head and tail of a cons cell into `hd` and `tl` respectively
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Split {
    pub hd: Register,
    pub tl: Register,
    pub list: Register,
}
encoding_impl!(Split, hd, tl, list);

/// Constructs a new closure, with an environment of size `arity`, pointing to `function`.
///
/// This instruction expects that `dest` is followed on the stack by `arity` registers, containing
/// the values being closed over for the closure environment. This is just like how arguments are
/// passed to the `Call` instruction, except there are no reserved registers between `dest` and the
/// first argument like there is with `Call`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Closure {
    pub dest: Register,
    pub arity: Arity,
    pub function: FunId,
}
encoding_impl!(Closure, dest, arity, function);

/// Extracts the value at `index` from the environment of `fun`, a closure, and places it in `dest`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnpackEnv {
    pub dest: Register,
    pub fun: Register,
    pub index: Arity,
}
encoding_impl!(UnpackEnv, dest, fun, index);

/// Constructs a new tuple with `arity` elements.
///
/// Like `Closure`, it is expected that there are `arity` registers immediately following
/// `dest` containing the values of the elements, in order.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Tuple {
    pub dest: Register,
    pub arity: Arity,
}
encoding_impl!(Tuple, dest, arity);

/// Constructs a new empty tuple, with capacity for `arity` elements.
///
/// There are no arguments expected on the stack for this op, unlike `Tuple`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TupleWithCapacity {
    pub dest: Register,
    pub arity: Arity,
}
encoding_impl!(TupleWithCapacity, dest, arity);

/// Loads the arity of `tuple` into `dest`.
///
/// NOTE: It should always be the case that `tuple` is actually a tuple, if it is not,
/// implementations are free to treat that as undefined behavior or panic.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TupleArity {
    pub dest: Register,
    pub tuple: Register,
}
encoding_impl!(TupleArity, dest, tuple);

/// Loads the element at `index` in `tuple` into `dest`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GetElement {
    pub dest: Register,
    pub tuple: Register,
    pub index: Arity,
}
encoding_impl!(GetElement, dest, tuple, index);

/// Creates a new copy of `tuple`, with the value at `index` replaced with `value`.
///
/// This conceptually clones `tuple`, i.e. the original is not modified in any way.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SetElement {
    pub dest: Register,
    pub tuple: Register,
    pub index: Arity,
    pub value: Register,
}
encoding_impl!(SetElement, dest, tuple, index, value);

/// Stores `value` at `index` in `tuple`.
///
/// Unlike `SetElement`, this op mutates the original tuple, and is only valid in contexts
/// where the original tuple has no other references. This is commonly used when constructing
/// tuples.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SetElementMut {
    pub tuple: Register,
    pub index: Arity,
    pub value: Register,
}
encoding_impl!(SetElementMut, tuple, index, value);

/// Constructs a new empty map with space for `capacity` key/value pairs.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Map {
    pub dest: Register,
    pub capacity: usize,
}
encoding_impl!(Map, dest, capacity);

/// Creates a new copy of `map` with `key` set to `value`.
///
/// This does not modify original map in any way.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapPut {
    pub dest: Register,
    pub map: Register,
    pub key: Register,
    pub value: Register,
}
encoding_impl!(MapPut, dest, map, key, value);

/// Sets `key` to `value` in `map`.
///
/// Unlike `MapPut`, this op mutates the original map, so is not safe to use in contexts
/// where there are other references to the original map.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapPutMut {
    pub map: Register,
    pub key: Register,
    pub value: Register,
}
encoding_impl!(MapPutMut, map, key, value);

/// Like `MapPut`, but raises a `badkey` error if `key` does not exist in `map`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapUpdate {
    pub dest: Register,
    pub map: Register,
    pub key: Register,
    pub value: Register,
}
encoding_impl!(MapUpdate, dest, map, key, value);

/// Like `MapPutMut`, but raises a `badkey` error if `key` does not exist in `map`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapUpdateMut {
    pub map: Register,
    pub key: Register,
    pub value: Register,
}
encoding_impl!(MapUpdateMut, map, key, value);

/// This is a fused instruction, semantically equivalent to `Map` plus one or more `MapPutMut` ops.
///
/// The benefit of this instruction is that it can be implemented much more efficiently, and avoid
/// any unnecessary reallocations of the underlying map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapExtendPut {
    pub dest: Register,
    pub map: Register,
    pub pairs: Vec<Register>,
}
encoding_impl!(MapExtendPut, dest, map, pairs);

/// This is like `MapExtendPut`, but with the semantics of `MapUpdateMut`, i.e. if any of the keys
/// in `pairs` do not exist in `map`, a badarg error must be raised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapExtendUpdate {
    pub dest: Register,
    pub map: Register,
    pub pairs: Vec<Register>,
}
encoding_impl!(MapExtendUpdate, dest, map, pairs);

/// This operation tests `map` for `key`, and if present, loads it into `value`.
///
/// If `map` does not contain `key`, `is_err` is set to true, otherwise false.
///
/// If `map` is not a map, implementations are free to treat that as undefined behavior or panic,
/// as the compiler should only generate this op after a type test ensuring that `map` is indeed
/// a map.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapTryGet {
    pub is_err: Register,
    pub value: Register,
    pub map: Register,
    pub key: Register,
}
encoding_impl!(MapTryGet, is_err, value, map, key);

/// Moves one register, `src`, into another, `dest`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Mov {
    pub dest: Register,
    pub src: Register,
}
encoding_impl!(Mov, dest, src);

/// Like `Mov`, but predicated on `cond`.
///
/// If `cond` is not a boolean value, implementations are free to treat that as undefined behavior
/// or panic.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Cmov {
    pub cond: Register,
    pub dest: Register,
    pub src: Register,
}
encoding_impl!(Cmov, cond, dest, src);

/// This instruction sets up a new catch handler for any exceptions occurring while it is live
/// on the stack. A catch handler is torn down when a matching `EndCatch` is reached, or during
/// unwinding of the stack when the landing pad of the handler is entered.
///
/// Immediately following every `Catch`, is a `LandingPad` instruction, which is where control
/// will be transferred if an exception occurs while this catch handler is active.
///
/// It is expected that implementations will skip over the `LandingPad` instruction during
/// execution.
///
/// This instruction is allocated a register, `cp`, to which the instruction offset of the landing
/// pad should be written. This should be used during unwinding to resume execution in the catch
/// handler.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Catch {
    /// The stack slot to use for storing the continuation pointer
    pub cp: Register,
}
encoding_impl!(Catch, cp);

/// Uninstalls the exception handler installed by a previous `Catch`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EndCatch;
encoding_impl!(EndCatch);

/// Loads the current exception kind and reason into `kind` and `reason` respectively.
/// The `trace` register holds the "raw" stack trace, i.e. it is not a valid term, and
/// the compiler generates code to reify it into an actual Erlang stack trace on demand.
///
/// When control transfers to this instruction, it is treated like a `Br`, except the
/// `kind`, `reason`, and `trace` values are passed as arguments to the destination block
///
/// NOTE: This instruction must be the first instruction following a `Catch`, and may only
/// occur following a `Catch`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LandingPad {
    pub kind: Register,
    pub reason: Register,
    pub trace: Register,
    pub offset: JumpOffset,
}
encoding_impl!(LandingPad, kind, reason, trace, offset);

/// Reifies a "raw" stack trace into an Erlang term representation of that trace.
///
/// This is done separately from `LandingPad`, as not all catch handlers use the stack trace,
/// and it is quite expensive to construct. This instruction allows us to only do so
/// when needed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StackTrace {
    pub dest: Register,
}
encoding_impl!(StackTrace, dest);

/// This instruction implements `erlang:!/2`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SendOp {
    pub recipient: Register,
    pub message: Register,
}
encoding_impl!(SendOp, recipient, message);

/// Peeks at the message pointed to by the active receive cursor in the mailbox, unless
/// that message has already been peeked.
///
/// If no receive cursor is active, one is initialized at the front of the queue, unless
/// there are no messages available.
///
/// Returns whether a message is available for examination, and the message itself if available.
/// This does not remove that message from the mailbox, that is left to `RecvPop` when a message
/// is matched by body of a `receive`. If the cursor currently points to a message we've already
/// seen, it is equivalent to there being no messages available.
///
/// This is the first instruction invoked in the implementation of `receive`, the following
/// pseudocode lays out how the various `receive` ops interoperate, where `is_match` represents
/// the generated code which determines if the currently available message matches any of the
/// patterns expressed in the `receive`.
///
/// ```text
/// loop:
///    let (available, message) = RecvPeek();
///    if available {
///       let (matched, body) = is_match(message);
///       if matched {
///         RecvPop();
///         body();
///         goto done;
///       } else {
///         RecvNext();
///         goto loop;
///       }
///    } else {
///       let timed_out = RecvWaitTimeout(timeout); # RecvWait and RecvTimeout are implicitly fused
///       if timed_out {
///         goto after;
///       } else {
///         goto loop;
///       }
///    }
/// after:
///    ...
/// done:
///    ...
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RecvPeek {
    pub available: Register,
    pub message: Register,
}
encoding_impl!(RecvPeek, available, message);

/// This instruction advances the receive cursor to the next message in the queue.
///
/// If there are no additional messages available, it leaves the cursor where it was.
///
/// This is called when the currently selected message was not matched by the `receive` body,
/// so we're moving on to the next message to see if it matches.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RecvNext;
encoding_impl!(RecvNext);

/// Suspends the current process until it receives more messages, or `timeout` expires.
///
/// This instruction sets up a process timer, and yields control to the scheduler. When
/// the process resumes, it will resume to the next instruction following this one, which
/// must always be a `RecvTimeout`. That instruction determines whether the `receive` timed
/// out or not.
///
/// * `dest` will be set to a boolean indicating whether or not the receive timed out
/// * `timeout` is the timeout value to use, may either be the atom `infinity` or an integer
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RecvWait {
    pub dest: Register,
    pub timeout: Register,
}
encoding_impl!(RecvWait, dest, timeout);

/// This instruction stores the "timed out" flag in `dest` when control resumes in a
/// suspended process due to either a new message in the mailbox, or expiration of a
/// process timer.
///
/// This instruction is split from `RecvWait` even though they appear to be fused
/// in terms of behavior, because they represent the entry and exit phases of the operation,
/// which are quite different.
///
/// NOTE: If resumed due to new messages, the process timer is _NOT_ cancelled, but is
/// instead kept alive until a message is "received" via `RecvPop`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RecvTimeout {
    pub dest: Register,
}
encoding_impl!(RecvTimeout, dest);

/// Removes the currently selected message from the mailbox
///
/// This operation conceptually represents "acceptance" of a message, i.e. until this
/// instruction is invoked on a message, that message remains in the queue indefinitely.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RecvPop;
encoding_impl!(RecvPop);

/// This is not present in generated code, but is instead used like `NormalExit` and `ContinueExit`,
/// i.e. it is stored at a constant location in the generated code, and when a native function is
/// called which is implemented as a generator, the instruction pointer is set to this instruction,
/// and the generator is stored in the process state. Each time this instruction is dispatched, it
/// loads the generator from the process state and resumes it, if it is not completed, it yields
/// back to the scheduler. If completed, it behaves like a function call, i.e. it places the result
/// of the generator in the return register, then dispatches `Ret` to resume execution where the
/// original call instruction left off.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Await;
encoding_impl!(Await);

/// This is used much like `Await`, in that it is a special instruction not generated by the
/// compiler which is used to trampoline from a natively-implemented function to a bytecode
/// function, or to implement a yielding BIF by trapping to itself. Conceptually trapping to
/// a function is semantically equivalent to a tail call, in that it reuses the current call
/// frame for the callee arguments, and returning from the trap function returns from the
/// original caller.
///
/// It is expected that when a function is trapping, that it stores the desired callee in the
/// process state.
///
/// NOTE: This is called "trap" to correspond to the equivalent naming/functionality in the BEAM,
/// but it is more appropriate to think of it as a "trampoline" instruction.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Trap;
encoding_impl!(Trap);

/// Yield control back to the scheduler
///
/// NOTE: Currently this is unused, but is available for situations in which we want manual control
/// over this.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Yield;
encoding_impl!(Yield);

/// Stops execution of the process and performs a garbage collection
///
/// By default all collections are minor, unless forced to be full sweeps, either during execution
/// of the collection, or by setting `fullsweep` to true here.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GarbageCollect {
    pub fullsweep: bool,
}
encoding_impl!(GarbageCollect, fullsweep);

/// Exits the current process normally
///
/// This instruction is the first non-nop instruction in every bytecode program, so by default if
/// control returns to the top of the stack, this instruction will be dispatched, terminating a
/// process normally.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NormalExit;
encoding_impl!(NormalExit);

/// Continues with exiting a process.
///
/// NOTE: This instruction is never built, but is used by the emulator internally.
///
/// Termination of a process can take time, depending on what system resources it has attached, and
/// what signals remain in its signal queue to handle. Exiting is thus broken up into phases, so
/// that we can yield to the scheduler periodically. Once exiting has begun, the instruction pointer
/// of a process is pinned to this instruction. This ensures that while the process is still
/// exiting, each time it gets scheduled it will resume with the next phase, always making progress.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ContinueExit;
encoding_impl!(ContinueExit);

/// Exits the current process, with `reason`.
///
/// This is semantically equivalent to `erlang:exit/1`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Exit1 {
    pub reason: Register,
}
encoding_impl!(Exit1, reason);

/// Exits a given pid, places `true` in `dest`.
///
/// If `pid` is the current process, this is equivalent to `Exit1`.
///
/// This is semantically equivalent to `erlang:exit/2`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Exit2 {
    pub dest: Register,
    pub pid: Register,
    pub reason: Register,
}
encoding_impl!(Exit2, dest, pid, reason);

/// Raises an error with `reason`.
///
/// This is semantically equivalent to `erlang:error/1`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Error1 {
    pub reason: Register,
}
encoding_impl!(Error1, reason);

/// Throws to the nearest catch handler with `reason`
///
/// This is semantically equivalent to `erlang:throw/1`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Throw1 {
    pub reason: Register,
}
encoding_impl!(Throw1, reason);

/// This is a more general form of exception primitive, which can be used to
/// raise or re-raise any of the exception types. It is semantically equivalent
/// to `erlang:raise/3`.
///
/// If the exception kind or the stack trace are invalid, the atom `badarg` is returned in `dest`.
///
/// Otherwise, this instruction behaves like one of the other exception primitives,
/// depending on the type of exception being raised.
///
/// * `dest` holds the return value if an invalid argument is given
/// * `kind` holds the kind of exception being raised
/// * `reason` holds the exception reason
/// * `trace` is optional, but holds the exception trace as an Erlang term
/// * `opts` is optional, and is only present when `trace` is present, and holds additional info for
///   the exception
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Raise {
    pub dest: Register,
    pub kind: Register,
    pub reason: Register,
    pub trace: Option<Register>,
    pub opts: Option<Register>,
}
encoding_impl!(Raise, dest, kind, reason, trace, opts);

/// Halts execution of the runtime with the given status/options
///
/// This is equivalent to `erlang:halt/1,2`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Halt {
    pub status: Register,
    pub options: Register,
}
encoding_impl!(Halt, status, options);

/// Initializes a new binary/bitstring builder
///
/// This is invoked when beginning construction of a new binary or bitstring.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BsInit {
    pub dest: Register,
}
encoding_impl!(BsInit, dest);

/// Finalizes an in-progress binary/bitstring, placing the binary/bitstring term in `dest`
///
/// This consumes the builder, which should free any resources it uses behind the scenes.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BsFinish {
    pub dest: Register,
    pub builder: Register,
}
encoding_impl!(BsFinish, dest, builder);

/// Pushes a value on to an in-progress binary builder
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BsPush {
    pub dest: Register,
    pub builder: Register,
    pub value: Register,
    pub size: Option<Register>,
    pub spec: BinaryEntrySpecifier,
}
encoding_impl!(BsPush, dest, builder, value, size, spec);

/// Tests that `bin` is a binary/bitstring value, and starts a new match context if so.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BsMatchStart {
    pub is_err: Register,
    pub context: Register,
    pub bin: Register,
}
encoding_impl!(BsMatchStart, is_err, context, bin);

/// Attempts to extract a value from the provided match context, using the given specification.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BsMatch {
    pub is_err: Register,
    pub value: Register,
    pub next: Register,
    pub context: Register,
    pub size: Option<Register>,
    pub spec: BinaryEntrySpecifier,
}
encoding_impl!(BsMatch, is_err, value, next, context, size, spec);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum BsMatchSkipType {
    BigUnsigned = 0,
    BigSigned,
    LittleUnsigned,
    LittleSigned,
    NativeUnsigned,
    NativeSigned,
}
impl TryFrom<u8> for BsMatchSkipType {
    type Error = ();

    fn try_from(n: u8) -> Result<Self, Self::Error> {
        match n {
            0 => Ok(Self::BigUnsigned),
            1 => Ok(Self::BigSigned),
            2 => Ok(Self::LittleUnsigned),
            3 => Ok(Self::LittleSigned),
            4 => Ok(Self::NativeUnsigned),
            5 => Ok(Self::NativeSigned),
            _ => Err(()),
        }
    }
}

/// Like `BsMatch`, but specialized for skipping over the matched region of the underlying
/// binary/bitstring, rather than extracting the matched value. Specifically, the matched value must
/// be an integer, using the provided specification. Behaves the same way as `BsMatch` if the match
/// fails.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BsMatchSkip {
    pub is_err: Register,
    pub next: Register,
    pub context: Register,
    // The specification of how to parse the integer value to match
    pub ty: BsMatchSkipType,
    pub size: Register,
    pub unit: u8,
    // The value to match against
    pub value: Register,
}
encoding_impl!(BsMatchSkip, is_err, next, context, ty, size, unit, value);

/// Tests that `size` bytes remain in the underlying input of the match context.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BsTestTail {
    pub dest: Register,
    pub context: Register,
    pub size: usize,
}
encoding_impl!(BsTestTail, dest, context, size);

/// The first instruction in every function, providing metadata about the function
///
/// This is largely used to set up stack frames on entry, specifically by allocating
/// space on the stack for all of the locals needed by the function.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FuncInfo {
    pub id: FunId,
    pub arity: Arity,
    pub frame_size: u16,
}
encoding_impl!(FuncInfo, id, arity, frame_size);

/// Puts the identity of the current process in `dest`
///
/// This is equivalent to `erlang:self/0`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Identity {
    pub dest: Register,
}
encoding_impl!(Identity, dest);

bitflags::bitflags! {
    pub struct SpawnOpts: u8 {
        const LINK = 1;
        const MONITOR = 1 << 1;
    }
}

/// Spawns a new process running `fun`, placing its pid in `dest`
///
/// This can represent all variants of `spawn`, `spawn_link`, `spawn_monitor` that take a closure as
/// an argument.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Spawn2 {
    pub dest: Register,
    pub fun: Register,
    pub opts: SpawnOpts,
}
encoding_impl!(Spawn2, dest, fun, opts);

/// Spawns a new process running `fun`, placing its pid in `dest`
///
/// This can represent all variants of `spawn`, `spawn_link`, `spawn_monitor` that take an MFA as an
/// argument.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Spawn3 {
    pub dest: Register,
    pub fun: FunId,
    pub args: Register,
    pub opts: SpawnOpts,
}
encoding_impl!(Spawn3, dest, fun, args, opts);

/// Same as `Spawn3`, but for a runtime-resolved MFA
///
/// This can represent all variants of `spawn`, `spawn_link`, `spawn_monitor` that take an MFA as an
/// argument.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Spawn3Indirect {
    pub dest: Register,
    pub module: Register,
    pub function: Register,
    pub args: Register,
    pub opts: SpawnOpts,
}
encoding_impl!(Spawn3Indirect, dest, module, function, args, opts);

impl<A: Atom> Opcode<A> {
    #[cfg(any(test, feature = "std"))]
    fn tag(&self) -> u8 {
        match self {
            Self::Nop(_) => 0,
            Self::Mov(_) => 1,
            Self::Cmov(_) => 2,
            Self::Ret(_) => 3,
            Self::Br(_) => 4,
            Self::Brz(_) => 5,
            Self::Brnz(_) => 6,
            Self::JumpTable(_) => 7,
            Self::JumpTableEntry(_) => 8,
            Self::Call(_) => 9,
            Self::CallApply2(_) => 10,
            Self::CallApply3(_) => 11,
            Self::CallNative(_) => 12,
            Self::CallStatic(_) => 13,
            Self::CallIndirect(_) => 14,
            Self::Enter(_) => 15,
            Self::EnterApply2(_) => 16,
            Self::EnterApply3(_) => 17,
            Self::EnterNative(_) => 18,
            Self::EnterStatic(_) => 19,
            Self::EnterIndirect(_) => 20,
            Self::IsAtom(_) => 21,
            Self::IsBool(_) => 22,
            Self::IsNil(_) => 23,
            Self::IsTuple(_) => 24,
            Self::IsTupleFetchArity(_) => 25,
            Self::IsMap(_) => 26,
            Self::IsCons(_) => 27,
            Self::IsList(_) => 28,
            Self::IsInt(_) => 29,
            Self::IsFloat(_) => 30,
            Self::IsNumber(_) => 31,
            Self::IsPid(_) => 32,
            Self::IsRef(_) => 33,
            Self::IsPort(_) => 34,
            Self::IsBinary(_) => 35,
            Self::IsFunction(_) => 36,
            Self::LoadNil(_) => 37,
            Self::LoadBool(_) => 38,
            Self::LoadAtom(_) => 39,
            Self::LoadInt(_) => 40,
            Self::LoadBig(_) => 41,
            Self::LoadFloat(_) => 42,
            Self::LoadBinary(_) => 43,
            Self::LoadBitstring(_) => 44,
            Self::Not(_) => 45,
            Self::And(_) => 46,
            Self::AndAlso(_) => 47,
            Self::Or(_) => 48,
            Self::OrElse(_) => 49,
            Self::Xor(_) => 50,
            Self::Bnot(_) => 51,
            Self::Band(_) => 52,
            Self::Bor(_) => 53,
            Self::Bxor(_) => 54,
            Self::Bsl(_) => 55,
            Self::Bsr(_) => 56,
            Self::Div(_) => 57,
            Self::Rem(_) => 58,
            Self::Neg(_) => 59,
            Self::Add(_) => 60,
            Self::Sub(_) => 61,
            Self::Mul(_) => 62,
            Self::Divide(_) => 63,
            Self::ListAppend(_) => 64,
            Self::ListRemove(_) => 65,
            Self::Eq(_) => 66,
            Self::Neq(_) => 67,
            Self::Gt(_) => 68,
            Self::Gte(_) => 69,
            Self::Lt(_) => 70,
            Self::Lte(_) => 71,
            Self::Cons(_) => 72,
            Self::Split(_) => 73,
            Self::Head(_) => 74,
            Self::Tail(_) => 75,
            Self::Closure(_) => 76,
            Self::UnpackEnv(_) => 77,
            Self::Tuple(_) => 78,
            Self::TupleWithCapacity(_) => 79,
            Self::TupleArity(_) => 80,
            Self::GetElement(_) => 81,
            Self::SetElement(_) => 82,
            Self::SetElementMut(_) => 83,
            Self::Map(_) => 84,
            Self::MapPut(_) => 85,
            Self::MapPutMut(_) => 86,
            Self::MapUpdate(_) => 87,
            Self::MapUpdateMut(_) => 88,
            Self::MapExtendPut(_) => 89,
            Self::MapExtendUpdate(_) => 90,
            Self::MapTryGet(_) => 91,
            Self::Catch(_) => 92,
            Self::EndCatch(_) => 93,
            Self::LandingPad(_) => 94,
            Self::StackTrace(_) => 95,
            Self::Raise(_) => 96,
            Self::Send(_) => 97,
            Self::RecvPeek(_) => 98,
            Self::RecvNext(_) => 99,
            Self::RecvWait(_) => 100,
            Self::RecvTimeout(_) => 101,
            Self::RecvPop(_) => 102,
            Self::Await(_) => 103,
            Self::Yield(_) => 104,
            Self::GarbageCollect(_) => 105,
            Self::NormalExit(_) => 106,
            Self::ContinueExit(_) => 107,
            Self::Exit1(_) => 108,
            Self::Exit2(_) => 109,
            Self::Error1(_) => 110,
            Self::Throw1(_) => 111,
            Self::Halt(_) => 112,
            Self::BsInit(_) => 113,
            Self::BsPush(_) => 114,
            Self::BsFinish(_) => 115,
            Self::BsMatchStart(_) => 116,
            Self::BsMatch(_) => 117,
            Self::BsMatchSkip(_) => 118,
            Self::BsTestTail(_) => 119,
            Self::FuncInfo(_) => 120,
            Self::Identity(_) => 121,
            Self::Spawn2(_) => 122,
            Self::Spawn3(_) => 123,
            Self::Spawn3Indirect(_) => 124,
            Self::Trap(_) => 125,
        }
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for Opcode<A> {
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        match reader.read_byte()? {
            0 => Ok(Self::Nop(Nop)),
            1 => Ok(Self::Mov(Decode::decode(reader)?)),
            2 => Ok(Self::Cmov(Decode::decode(reader)?)),
            3 => Ok(Self::Ret(Decode::decode(reader)?)),
            4 => Ok(Self::Br(Decode::decode(reader)?)),
            5 => Ok(Self::Brz(Decode::decode(reader)?)),
            6 => Ok(Self::Brnz(Decode::decode(reader)?)),
            7 => Ok(Self::JumpTable(Decode::decode(reader)?)),
            8 => Ok(Self::JumpTableEntry(Decode::decode(reader)?)),
            9 => Ok(Self::Call(Decode::decode(reader)?)),
            10 => Ok(Self::CallApply2(Decode::decode(reader)?)),
            11 => Ok(Self::CallApply3(Decode::decode(reader)?)),
            12 => Ok(Self::CallNative(Decode::decode(reader)?)),
            13 => Ok(Self::CallStatic(Decode::decode(reader)?)),
            14 => Ok(Self::CallIndirect(Decode::decode(reader)?)),
            15 => Ok(Self::Enter(Decode::decode(reader)?)),
            16 => Ok(Self::EnterApply2(Decode::decode(reader)?)),
            17 => Ok(Self::EnterApply3(Decode::decode(reader)?)),
            18 => Ok(Self::EnterNative(Decode::decode(reader)?)),
            19 => Ok(Self::EnterStatic(Decode::decode(reader)?)),
            20 => Ok(Self::EnterIndirect(Decode::decode(reader)?)),
            21 => Ok(Self::IsAtom(Decode::decode(reader)?)),
            22 => Ok(Self::IsBool(Decode::decode(reader)?)),
            23 => Ok(Self::IsNil(Decode::decode(reader)?)),
            24 => Ok(Self::IsTuple(Decode::decode(reader)?)),
            25 => Ok(Self::IsTupleFetchArity(Decode::decode(reader)?)),
            26 => Ok(Self::IsMap(Decode::decode(reader)?)),
            27 => Ok(Self::IsCons(Decode::decode(reader)?)),
            28 => Ok(Self::IsList(Decode::decode(reader)?)),
            29 => Ok(Self::IsInt(Decode::decode(reader)?)),
            30 => Ok(Self::IsFloat(Decode::decode(reader)?)),
            31 => Ok(Self::IsNumber(Decode::decode(reader)?)),
            32 => Ok(Self::IsPid(Decode::decode(reader)?)),
            33 => Ok(Self::IsRef(Decode::decode(reader)?)),
            34 => Ok(Self::IsPort(Decode::decode(reader)?)),
            35 => Ok(Self::IsBinary(Decode::decode(reader)?)),
            36 => Ok(Self::IsFunction(Decode::decode(reader)?)),
            37 => Ok(Self::LoadNil(Decode::decode(reader)?)),
            38 => Ok(Self::LoadBool(Decode::decode(reader)?)),
            39 => Ok(Self::LoadAtom(Decode::decode(reader)?)),
            40 => Ok(Self::LoadInt(Decode::decode(reader)?)),
            41 => Ok(Self::LoadBig(Decode::decode(reader)?)),
            42 => Ok(Self::LoadFloat(Decode::decode(reader)?)),
            43 => Ok(Self::LoadBinary(Decode::decode(reader)?)),
            44 => Ok(Self::LoadBitstring(Decode::decode(reader)?)),
            45 => Ok(Self::Not(Decode::decode(reader)?)),
            46 => Ok(Self::And(Decode::decode(reader)?)),
            47 => Ok(Self::AndAlso(Decode::decode(reader)?)),
            48 => Ok(Self::Or(Decode::decode(reader)?)),
            49 => Ok(Self::OrElse(Decode::decode(reader)?)),
            50 => Ok(Self::Xor(Decode::decode(reader)?)),
            51 => Ok(Self::Bnot(Decode::decode(reader)?)),
            52 => Ok(Self::Band(Decode::decode(reader)?)),
            53 => Ok(Self::Bor(Decode::decode(reader)?)),
            54 => Ok(Self::Bxor(Decode::decode(reader)?)),
            55 => Ok(Self::Bsl(Decode::decode(reader)?)),
            56 => Ok(Self::Bsr(Decode::decode(reader)?)),
            57 => Ok(Self::Div(Decode::decode(reader)?)),
            58 => Ok(Self::Rem(Decode::decode(reader)?)),
            59 => Ok(Self::Neg(Decode::decode(reader)?)),
            60 => Ok(Self::Add(Decode::decode(reader)?)),
            61 => Ok(Self::Sub(Decode::decode(reader)?)),
            62 => Ok(Self::Mul(Decode::decode(reader)?)),
            63 => Ok(Self::Divide(Decode::decode(reader)?)),
            64 => Ok(Self::ListAppend(Decode::decode(reader)?)),
            65 => Ok(Self::ListRemove(Decode::decode(reader)?)),
            66 => Ok(Self::Eq(Decode::decode(reader)?)),
            67 => Ok(Self::Neq(Decode::decode(reader)?)),
            68 => Ok(Self::Gt(Decode::decode(reader)?)),
            69 => Ok(Self::Gte(Decode::decode(reader)?)),
            70 => Ok(Self::Lt(Decode::decode(reader)?)),
            71 => Ok(Self::Lte(Decode::decode(reader)?)),
            72 => Ok(Self::Cons(Decode::decode(reader)?)),
            73 => Ok(Self::Split(Decode::decode(reader)?)),
            74 => Ok(Self::Head(Decode::decode(reader)?)),
            75 => Ok(Self::Tail(Decode::decode(reader)?)),
            76 => Ok(Self::Closure(Decode::decode(reader)?)),
            77 => Ok(Self::UnpackEnv(Decode::decode(reader)?)),
            78 => Ok(Self::Tuple(Decode::decode(reader)?)),
            79 => Ok(Self::TupleWithCapacity(Decode::decode(reader)?)),
            80 => Ok(Self::TupleArity(Decode::decode(reader)?)),
            81 => Ok(Self::GetElement(Decode::decode(reader)?)),
            82 => Ok(Self::SetElement(Decode::decode(reader)?)),
            83 => Ok(Self::SetElementMut(Decode::decode(reader)?)),
            84 => Ok(Self::Map(Decode::decode(reader)?)),
            85 => Ok(Self::MapPut(Decode::decode(reader)?)),
            86 => Ok(Self::MapPutMut(Decode::decode(reader)?)),
            87 => Ok(Self::MapUpdate(Decode::decode(reader)?)),
            88 => Ok(Self::MapUpdateMut(Decode::decode(reader)?)),
            89 => Ok(Self::MapExtendPut(Decode::decode(reader)?)),
            90 => Ok(Self::MapExtendUpdate(Decode::decode(reader)?)),
            91 => Ok(Self::MapTryGet(Decode::decode(reader)?)),
            92 => Ok(Self::Catch(Decode::decode(reader)?)),
            93 => Ok(Self::EndCatch(EndCatch)),
            94 => Ok(Self::LandingPad(Decode::decode(reader)?)),
            95 => Ok(Self::StackTrace(Decode::decode(reader)?)),
            96 => Ok(Self::Raise(Decode::decode(reader)?)),
            97 => Ok(Self::Send(Decode::decode(reader)?)),
            98 => Ok(Self::RecvPeek(Decode::decode(reader)?)),
            99 => Ok(Self::RecvNext(RecvNext)),
            100 => Ok(Self::RecvWait(Decode::decode(reader)?)),
            101 => Ok(Self::RecvTimeout(Decode::decode(reader)?)),
            102 => Ok(Self::RecvPop(RecvPop)),
            103 => Ok(Self::Await(Await)),
            104 => Ok(Self::Yield(Yield)),
            105 => Ok(Self::GarbageCollect(Decode::decode(reader)?)),
            106 => Ok(Self::NormalExit(Decode::decode(reader)?)),
            107 => Ok(Self::ContinueExit(Decode::decode(reader)?)),
            108 => Ok(Self::Exit1(Decode::decode(reader)?)),
            109 => Ok(Self::Exit2(Decode::decode(reader)?)),
            110 => Ok(Self::Error1(Decode::decode(reader)?)),
            111 => Ok(Self::Throw1(Decode::decode(reader)?)),
            112 => Ok(Self::Halt(Decode::decode(reader)?)),
            113 => Ok(Self::BsInit(Decode::decode(reader)?)),
            114 => Ok(Self::BsPush(Decode::decode(reader)?)),
            115 => Ok(Self::BsFinish(Decode::decode(reader)?)),
            116 => Ok(Self::BsMatchStart(Decode::decode(reader)?)),
            117 => Ok(Self::BsMatch(Decode::decode(reader)?)),
            118 => Ok(Self::BsMatchSkip(Decode::decode(reader)?)),
            119 => Ok(Self::BsTestTail(Decode::decode(reader)?)),
            120 => Ok(Self::FuncInfo(Decode::decode(reader)?)),
            121 => Ok(Self::Identity(Decode::decode(reader)?)),
            122 => Ok(Self::Spawn2(Decode::decode(reader)?)),
            123 => Ok(Self::Spawn3(Decode::decode(reader)?)),
            124 => Ok(Self::Spawn3Indirect(Decode::decode(reader)?)),
            125 => Ok(Self::Trap(Trap)),
            _ => Err(ReadError::Invalid),
        }
    }
}

#[cfg(any(test, feature = "std"))]
impl<A: Atom> Encode<A> for Opcode<A> {
    fn encode(&self, writer: &mut BytecodeWriter<A>) -> std::io::Result<()> {
        writer.write_byte(self.tag())?;
        match self {
            Self::Nop(op) => op.encode(writer),
            Self::Mov(op) => op.encode(writer),
            Self::Cmov(op) => op.encode(writer),
            Self::Ret(op) => op.encode(writer),
            Self::Br(op) => op.encode(writer),
            Self::Brz(op) => op.encode(writer),
            Self::Brnz(op) => op.encode(writer),
            Self::JumpTable(op) => op.encode(writer),
            Self::JumpTableEntry(op) => op.encode(writer),
            Self::Call(op) => op.encode(writer),
            Self::CallApply2(op) => op.encode(writer),
            Self::CallApply3(op) => op.encode(writer),
            Self::CallNative(op) => op.encode(writer),
            Self::CallStatic(op) => op.encode(writer),
            Self::CallIndirect(op) => op.encode(writer),
            Self::Enter(op) => op.encode(writer),
            Self::EnterApply2(op) => op.encode(writer),
            Self::EnterApply3(op) => op.encode(writer),
            Self::EnterNative(op) => op.encode(writer),
            Self::EnterStatic(op) => op.encode(writer),
            Self::EnterIndirect(op) => op.encode(writer),
            Self::IsAtom(op) => op.encode(writer),
            Self::IsBool(op) => op.encode(writer),
            Self::IsNil(op) => op.encode(writer),
            Self::IsTuple(op) => op.encode(writer),
            Self::IsTupleFetchArity(op) => op.encode(writer),
            Self::IsMap(op) => op.encode(writer),
            Self::IsCons(op) => op.encode(writer),
            Self::IsList(op) => op.encode(writer),
            Self::IsInt(op) => op.encode(writer),
            Self::IsFloat(op) => op.encode(writer),
            Self::IsNumber(op) => op.encode(writer),
            Self::IsPid(op) => op.encode(writer),
            Self::IsRef(op) => op.encode(writer),
            Self::IsPort(op) => op.encode(writer),
            Self::IsBinary(op) => op.encode(writer),
            Self::IsFunction(op) => op.encode(writer),
            Self::LoadNil(op) => op.encode(writer),
            Self::LoadBool(op) => op.encode(writer),
            Self::LoadAtom(op) => op.encode(writer),
            Self::LoadInt(op) => op.encode(writer),
            Self::LoadBig(op) => op.encode(writer),
            Self::LoadFloat(op) => op.encode(writer),
            Self::LoadBinary(op) => op.encode(writer),
            Self::LoadBitstring(op) => op.encode(writer),
            Self::Not(op) => op.encode(writer),
            Self::And(op) => op.encode(writer),
            Self::AndAlso(op) => op.encode(writer),
            Self::Or(op) => op.encode(writer),
            Self::OrElse(op) => op.encode(writer),
            Self::Xor(op) => op.encode(writer),
            Self::Bnot(op) => op.encode(writer),
            Self::Band(op) => op.encode(writer),
            Self::Bor(op) => op.encode(writer),
            Self::Bxor(op) => op.encode(writer),
            Self::Bsl(op) => op.encode(writer),
            Self::Bsr(op) => op.encode(writer),
            Self::Div(op) => op.encode(writer),
            Self::Rem(op) => op.encode(writer),
            Self::Neg(op) => op.encode(writer),
            Self::Add(op) => op.encode(writer),
            Self::Sub(op) => op.encode(writer),
            Self::Mul(op) => op.encode(writer),
            Self::Divide(op) => op.encode(writer),
            Self::ListAppend(op) => op.encode(writer),
            Self::ListRemove(op) => op.encode(writer),
            Self::Eq(op) => op.encode(writer),
            Self::Neq(op) => op.encode(writer),
            Self::Gt(op) => op.encode(writer),
            Self::Gte(op) => op.encode(writer),
            Self::Lt(op) => op.encode(writer),
            Self::Lte(op) => op.encode(writer),
            Self::Cons(op) => op.encode(writer),
            Self::Split(op) => op.encode(writer),
            Self::Head(op) => op.encode(writer),
            Self::Tail(op) => op.encode(writer),
            Self::Closure(op) => op.encode(writer),
            Self::UnpackEnv(op) => op.encode(writer),
            Self::Tuple(op) => op.encode(writer),
            Self::TupleWithCapacity(op) => op.encode(writer),
            Self::TupleArity(op) => op.encode(writer),
            Self::GetElement(op) => op.encode(writer),
            Self::SetElement(op) => op.encode(writer),
            Self::SetElementMut(op) => op.encode(writer),
            Self::Map(op) => op.encode(writer),
            Self::MapPut(op) => op.encode(writer),
            Self::MapPutMut(op) => op.encode(writer),
            Self::MapUpdate(op) => op.encode(writer),
            Self::MapUpdateMut(op) => op.encode(writer),
            Self::MapExtendPut(op) => op.encode(writer),
            Self::MapExtendUpdate(op) => op.encode(writer),
            Self::MapTryGet(op) => op.encode(writer),
            Self::Catch(op) => op.encode(writer),
            Self::EndCatch(op) => op.encode(writer),
            Self::LandingPad(op) => op.encode(writer),
            Self::StackTrace(op) => op.encode(writer),
            Self::Raise(op) => op.encode(writer),
            Self::Send(op) => op.encode(writer),
            Self::RecvPeek(op) => op.encode(writer),
            Self::RecvNext(op) => op.encode(writer),
            Self::RecvWait(op) => op.encode(writer),
            Self::RecvTimeout(op) => op.encode(writer),
            Self::RecvPop(op) => op.encode(writer),
            Self::Await(op) => op.encode(writer),
            Self::Yield(op) => op.encode(writer),
            Self::GarbageCollect(op) => op.encode(writer),
            Self::NormalExit(op) => op.encode(writer),
            Self::ContinueExit(op) => op.encode(writer),
            Self::Exit1(op) => op.encode(writer),
            Self::Exit2(op) => op.encode(writer),
            Self::Error1(op) => op.encode(writer),
            Self::Throw1(op) => op.encode(writer),
            Self::Halt(op) => op.encode(writer),
            Self::BsInit(op) => op.encode(writer),
            Self::BsPush(op) => op.encode(writer),
            Self::BsFinish(op) => op.encode(writer),
            Self::BsMatchStart(op) => op.encode(writer),
            Self::BsMatch(op) => op.encode(writer),
            Self::BsMatchSkip(op) => op.encode(writer),
            Self::BsTestTail(op) => op.encode(writer),
            Self::FuncInfo(op) => op.encode(writer),
            Self::Identity(op) => op.encode(writer),
            Self::Spawn2(op) => op.encode(writer),
            Self::Spawn3(op) => op.encode(writer),
            Self::Spawn3Indirect(op) => op.encode(writer),
            Self::Trap(op) => op.encode(writer),
        }
    }
}
