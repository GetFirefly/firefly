use alloc::vec::Vec;
use core::mem;
use core::num::NonZeroU32;

use firefly_binary::{BinaryEntrySpecifier, Endianness};
use firefly_number::BigInt;

use crate::reader::{eof_to_invalid, Decode};
#[cfg(any(test, feature = "std"))]
use crate::writer::Encode;

use super::*;

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
    Breq(Breq),
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
    Yield(Yield),
    GarbageCollect(GarbageCollect),
    // set reason to normmal
    // erts_exit_do_process(atoms::Normal)
    // schedule out
    NormalExit(NormalExit),
    // erts_continue_exit_process
    // schedule out
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
        assert_ne!(*self, Some(u8::MAX));
        writer.write_integer((*self).unwrap_or(u8::MAX))
    }
}

impl<A: Atom, T: AtomTable<Atom = A>> Decode<A, T> for Option<Register> {
    #[inline(always)]
    fn decode(reader: &mut BytecodeReader<A, T>) -> Result<Self, ReadError<T>> {
        reader
            .read_integer()
            .map(|n| if n == u8::MAX { None } else { Some(n) })
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Nop;
encoding_impl!(Nop);

/// Return the value in `reg`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Ret {
    pub reg: Register,
}
encoding_impl!(Ret, reg);

/// Unconditional jump
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Br {
    pub offset: JumpOffset,
}
encoding_impl!(Br, offset);

/// Jump if `reg` is a falsey value
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Brz {
    pub reg: Register,
    pub offset: JumpOffset,
}
encoding_impl!(Brz, reg, offset);

/// Jump if `reg` is a truthy value
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Brnz {
    pub reg: Register,
    pub offset: JumpOffset,
}
encoding_impl!(Brnz, reg, offset);

/// Jump if `reg` is equal to a given immediate integer
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Breq {
    pub reg: Register,
    pub imm: u32,
    pub offset: JumpOffset,
}
encoding_impl!(Breq, reg, imm, offset);

/// Jump to the instruction at `offset` (absolute) in a new stack frame, using `dest` for the return value
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Call {
    pub dest: Register,
    pub offset: usize,
}
encoding_impl!(Call, dest, offset);

/// Dynamic indirect function application in a new stack frame
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CallApply2 {
    pub dest: Register,
    pub callee: Register,
    pub argv: Register,
}
encoding_impl!(CallApply2, dest, callee, argv);

/// Dynamic function application to a static target in a new stack frame
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CallApply3 {
    pub dest: Register,
    pub module: Register,
    pub function: Register,
    pub argv: Register,
}
encoding_impl!(CallApply3, dest, module, function, argv);

/// Call `callee` as a natively-implemented function
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CallNative {
    pub dest: Register,
    pub callee: *const (),
    pub arity: Arity,
}
encoding_impl!(CallNative, dest, callee, arity);

/// Call `callee` in a new stack frame, using `dest` for the return value
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CallStatic {
    pub dest: Register,
    pub callee: FunId,
    pub arity: Arity,
}
encoding_impl!(CallStatic, dest, callee, arity);

/// Transfer control to `callee` in a new stack frame, using `dest` for the return value
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CallIndirect {
    pub dest: Register,
    pub callee: Register,
    pub arity: Arity,
}
encoding_impl!(CallIndirect, dest, callee, arity);

/// Transfer control to absolute `offset` without a new stack frame
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Enter {
    pub offset: usize,
}
encoding_impl!(Enter, offset);

/// Dynamic indirect function application without a new stack frame
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EnterApply2 {
    pub callee: Register,
    pub argv: Register,
}
encoding_impl!(EnterApply2, callee, argv);

/// Dynamic function application to a static target without a new stack frame
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EnterApply3 {
    pub module: Register,
    pub function: Register,
    pub argv: Register,
}
encoding_impl!(EnterApply3, module, function, argv);

/// Call `callee` as a natively-implemented function
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EnterNative {
    pub callee: *const (),
    pub arity: Arity,
}
encoding_impl!(EnterNative, callee, arity);

/// Transfer control to `callee` without a new stack frame
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EnterStatic {
    pub callee: FunId,
    pub arity: Arity,
}
encoding_impl!(EnterStatic, callee, arity);

/// Transfer control to `callee` without a new stack frame
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EnterIndirect {
    pub callee: Register,
    pub arity: Arity,
}
encoding_impl!(EnterIndirect, callee, arity);

/// Tests `value` to see if it is an atom, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsAtom {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsAtom, dest, value);

/// Tests `value` to see if it is a boolean, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsBool {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsBool, dest, value);

/// Tests `value` for nil, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsNil {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsNil, dest, value);

/// Tests `value` for a tuple, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsTuple {
    pub dest: Register,
    pub value: Register,
    pub arity: Option<NonZeroU32>,
}
encoding_impl!(IsTuple, dest, value, arity);

/// Tests `value` for a tuple, and puts the result in `dest`; if the term is a tuple, the arity is put in `arity`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsTupleFetchArity {
    pub dest: Register,
    pub arity: Register,
    pub value: Register,
}
encoding_impl!(IsTupleFetchArity, dest, arity, value);

/// Tests `value` for a map, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsMap {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsMap, dest, value);

/// Tests `value` for a non-empty list, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsCons {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsCons, dest, value);

/// Tests `value` for a list, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsList {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsList, dest, value);

/// Tests `value` for an integer, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsInt {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsInt, dest, value);

/// Tests `value` for a float, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsFloat {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsFloat, dest, value);

/// Tests `value` for any numeric value, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsNumber {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsNumber, dest, value);

/// Tests `value` for a pid, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsPid {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsPid, dest, value);

/// Tests `value` for a reference, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsRef {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsRef, dest, value);

/// Tests `value` for a port, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsPort {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsPort, dest, value);

/// Tests `value` for a binary/bitstring (depending on unit), and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsBinary {
    pub dest: Register,
    pub value: Register,
    pub unit: u8,
}
encoding_impl!(IsBinary, dest, value, unit);

/// Tests `value` for a function, and puts the result in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IsFunction {
    pub dest: Register,
    pub value: Register,
}
encoding_impl!(IsFunction, dest, value);

/// Load the nil value into `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LoadNil {
    pub dest: Register,
}
encoding_impl!(LoadNil, dest);

/// Load a boolean value into `dest`, opcode is implicitly followed by a single u8/bool value
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LoadBool {
    pub dest: Register,
    pub value: bool,
}
encoding_impl!(LoadBool, dest, value);

/// Load an atom value into `dest`, opcode is implicitly followed by a usize, representing the number of utf8-encoded bytes
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LoadAtom<A: Atom> {
    pub dest: Register,
    pub value: A,
}
encoding_impl!(LoadAtom<A>, dest, value);

/// Load a small integer value into `dest`, opcode is implicitly followed by an i64 value
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LoadInt {
    pub dest: Register,
    pub value: i64,
}
encoding_impl!(LoadInt, dest, value);

/// Load a big integer value into `dest`, opcode is implicitly followed by a usize, representing the number of big-endian encoded bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadBig {
    pub dest: Register,
    pub value: BigInt,
}
encoding_impl!(LoadBig, dest, value);

/// Load a floating point value into `dest`, opcode is implicitly followed by an f64 value
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct LoadFloat {
    pub dest: Register,
    pub value: f64,
}
encoding_impl!(LoadFloat, dest, value);

/// Load a binary value into `dest`, opcode is implicitly followed by a usize, representing the number of bytes
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LoadBinary {
    pub dest: Register,
    pub value: *const BinaryData,
}
encoding_impl!(LoadBinary, dest, value);

/// Load a bitstring value into `dest`, opcode is implicitly followed by a usize, representing the number of bits
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LoadBitstring {
    pub dest: Register,
    pub value: *const BinaryData,
}
encoding_impl!(LoadBitstring, dest, value);

/// Invert the boolean `cond` and place it in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Not {
    pub dest: Register,
    pub cond: Register,
}
encoding_impl!(Not, dest, cond);

/// Place the logical 'AND' of `lhs` and `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct And {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(And, dest, lhs, rhs);

/// Place the short-circuiting logical 'AND' of `lhs` and `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AndAlso {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(AndAlso, dest, lhs, rhs);

/// Place the logical 'OR' of `lhs` and `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Or {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Or, dest, lhs, rhs);

/// Place the short-circuiting logical 'OR' of `lhs` and `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct OrElse {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(OrElse, dest, lhs, rhs);

/// Place the logical 'XOR' of `lhs` and `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Xor {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Xor, dest, lhs, rhs);

/// Place the bitwise 'NOT' of `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bnot {
    pub dest: Register,
    pub rhs: Register,
}
encoding_impl!(Bnot, dest, rhs);

/// Place the bitwise 'AND' of `lhs` and `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Band {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Band, dest, lhs, rhs);

/// Place the bitwise 'OR' of `lhs` and `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bor {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Bor, dest, lhs, rhs);

/// Place the bitwise 'XOR' of `lhs` and `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bxor {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Bxor, dest, lhs, rhs);

/// Place the arithmetic bitshift-left of `value` by `shift` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bsl {
    pub dest: Register,
    pub value: Register,
    pub shift: Register,
}
encoding_impl!(Bsl, dest, value, shift);

/// Place the bitshift-right of `value` by `shift` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bsr {
    pub dest: Register,
    pub value: Register,
    pub shift: Register,
}
encoding_impl!(Bsr, dest, value, shift);

/// Place the result of dividing the integer `value` by `divisor` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Div {
    pub dest: Register,
    pub value: Register,
    pub divisor: Register,
}
encoding_impl!(Div, dest, value, divisor);

/// Place the remainder of dividing the integer `value` by `divisor` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Rem {
    pub dest: Register,
    pub value: Register,
    pub divisor: Register,
}
encoding_impl!(Rem, dest, value, divisor);

/// Place the result of negating `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Neg {
    pub dest: Register,
    pub rhs: Register,
}
encoding_impl!(Neg, dest, rhs);

/// Place the result of adding `lhs` and `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Add {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Add, dest, lhs, rhs);

/// Place the result of subtracting `lhs` and `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Sub {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Sub, dest, lhs, rhs);

/// Place the result of multiplying `lhs` and `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Mul {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Mul, dest, lhs, rhs);

/// Place the result of dividing `lhs` and `rhs` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Divide {
    pub dest: Register,
    pub lhs: Register,
    pub rhs: Register,
}
encoding_impl!(Divide, dest, lhs, rhs);

/// Place the result of appending `rhs` to `list` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ListAppend {
    pub dest: Register,
    pub list: Register,
    pub rhs: Register,
}
encoding_impl!(ListAppend, dest, list, rhs);

/// Place the result of removing `rhs` from `list` in `dest`
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

/// Create a cons cell in `dest` from `head` and `tail`
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

/// Create a closure with an env of `arity` size, pointing to `function`, implicitly followed by `arity` registers
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Closure {
    pub dest: Register,
    pub arity: Arity,
    pub function: FunId,
}
encoding_impl!(Closure, dest, arity, function);

/// Extracts a value at `index` from the environment of `fun`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnpackEnv {
    pub dest: Register,
    pub fun: Register,
    pub index: Arity,
}
encoding_impl!(UnpackEnv, dest, fun, index);

/// Create a tuple of `arity` size, implicitly followed by `arity` registers
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Tuple {
    pub dest: Register,
    pub arity: Arity,
}
encoding_impl!(Tuple, dest, arity);

/// Create an empty tuple of `arity` size
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TupleWithCapacity {
    pub dest: Register,
    pub arity: Arity,
}
encoding_impl!(TupleWithCapacity, dest, arity);

/// Get the arity of `tuple` and put it in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TupleArity {
    pub dest: Register,
    pub tuple: Register,
}
encoding_impl!(TupleArity, dest, tuple);

/// Get the element at `index` from `tuple` and put it in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GetElement {
    pub dest: Register,
    pub tuple: Register,
    pub index: Arity,
}
encoding_impl!(GetElement, dest, tuple, index);

/// dest = tuple[index] = value
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SetElement {
    pub dest: Register,
    pub tuple: Register,
    pub index: Arity,
    pub value: Register,
}
encoding_impl!(SetElement, dest, tuple, index, value);

/// tuple[index] = value
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SetElementMut {
    pub tuple: Register,
    pub index: Arity,
    pub value: Register,
}
encoding_impl!(SetElementMut, tuple, index, value);

/// Create an empty map
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Map {
    pub dest: Register,
    pub capacity: usize,
}
encoding_impl!(Map, dest, capacity);

/// dest = map#{key := value}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapPut {
    pub dest: Register,
    pub map: Register,
    pub key: Register,
    pub value: Register,
}
encoding_impl!(MapPut, dest, map, key, value);

/// map#{key := value}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapPutMut {
    pub map: Register,
    pub key: Register,
    pub value: Register,
}
encoding_impl!(MapPutMut, map, key, value);

/// dest = map#{key => value}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapUpdate {
    pub dest: Register,
    pub map: Register,
    pub key: Register,
    pub value: Register,
}
encoding_impl!(MapUpdate, dest, map, key, value);

/// map#{key => value}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapUpdateMut {
    pub map: Register,
    pub key: Register,
    pub value: Register,
}
encoding_impl!(MapUpdateMut, map, key, value);

/// dest = map#{key := value, .., keyN := valueN}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapExtendPut {
    pub dest: Register,
    pub map: Register,
    pub pairs: Vec<Register>,
}
encoding_impl!(MapExtendPut, dest, map, pairs);

/// dest = map#{key => value, .., keyN => valueN}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapExtendUpdate {
    pub dest: Register,
    pub map: Register,
    pub pairs: Vec<Register>,
}
encoding_impl!(MapExtendUpdate, dest, map, pairs);

/// is_err, value = map[key]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapTryGet {
    pub is_err: Register,
    pub value: Register,
    pub map: Register,
    pub key: Register,
}
encoding_impl!(MapTryGet, is_err, value, map, key);

/// Move `src` into `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Mov {
    pub dest: Register,
    pub src: Register,
}
encoding_impl!(Mov, dest, src);

/// Conditionally move `src` into `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Cmov {
    pub cond: Register,
    pub dest: Register,
    pub src: Register,
}
encoding_impl!(Cmov, cond, dest, src);

/// Indicates that the next instruction is the landing pad for any exceptions that occur until `EndCatch`
/// when encountered during evaluation, an exception handler is installed for all further instructions
/// until the handler is uninstalled by `EndCatch`
///
/// When this instruction is encountered, the instruction pointer is incremented to skip over the next
/// instruction, which must be a `LandingPad`, which describes where to transfer control if the catch
/// catches an unwind
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
/// The `trace` register is used as a token to fetch the stacktrace on demand, rather
/// than reifying it by default.
///
/// When control transfers to this instruction, it is treated like a `Br`, except the
/// `kind` and `reason` values are passed as arguments to the destination block
///
/// NOTE: must be the first instruction following a `Catch`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LandingPad {
    pub kind: Register,
    pub reason: Register,
    pub trace: Register,
    pub offset: JumpOffset,
}
encoding_impl!(LandingPad, kind, reason, trace, offset);

/// Reifies the current exception's stack trace as a term and stores it in `dest`
/// this is a separate instruction from `LandingPad` because the trace is not always
/// used, so we avoid constructing the trace unless explicitly requested
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StackTrace {
    pub dest: Register,
}
encoding_impl!(StackTrace, dest);

/// Sends `message` to `recipient`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SendOp {
    pub recipient: Register,
    pub message: Register,
}
encoding_impl!(SendOp, recipient, message);

/// Peeks at the currently selected message in the mailbox
///
/// `available` will be set to a boolean value indicating whether a message is available
/// `message` will be set to the none value if `available` is false, otherwise it will be
/// the currently selected message
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RecvPeek {
    pub available: Register,
    pub message: Register,
}
encoding_impl!(RecvPeek, available, message);

/// Selects the next message in the mailbox
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RecvNext;
encoding_impl!(RecvNext);

/// Blocks execution until a message is received or `timeout` expires
///
/// `dest` will be set to a boolean indicating whether or not the receive timed out
/// `timeout` is the timeout value to use, may either be the atom `infinity` or an integer
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RecvWait {
    pub dest: Register,
    pub timeout: Register,
}
encoding_impl!(RecvWait, dest, timeout);

/// This instruction is placed after a `RecvWait`, and control falls through to this
/// when the timeout of a `RecvWait` occurs. It has the same `dest` as its preceding
/// `RecvWait`, and sets the value to true if a timeout occurs
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RecvTimeout {
    pub dest: Register,
}
encoding_impl!(RecvTimeout, dest);

/// Removes the currently selected message from the mailbox
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RecvPop;
encoding_impl!(RecvPop);

/// Yield control back to the scheduler
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Yield;
encoding_impl!(Yield);

/// Stops execution of the process and performs a garbage collection
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GarbageCollect {
    pub fullsweep: bool,
}
encoding_impl!(GarbageCollect, fullsweep);

/// Exits the current process normally
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NormalExit;
encoding_impl!(NormalExit);

/// Continues with exiting a process.
///
/// This instruction is never built, but is used by the emulator internally
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ContinueExit;
encoding_impl!(ContinueExit);

/// Exits the current process
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Exit1 {
    pub reason: Register,
}
encoding_impl!(Exit1, reason);

/// Exits a given pid, places `true` in `dest`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Exit2 {
    pub dest: Register,
    pub pid: Register,
    pub reason: Register,
}
encoding_impl!(Exit2, dest, pid, reason);

/// Raises an error with the given reason
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Error1 {
    pub reason: Register,
}
encoding_impl!(Error1, reason);

/// Throws to the nearest catch handler with the given reason
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Throw1 {
    pub reason: Register,
}
encoding_impl!(Throw1, reason);

/// Generates a user-defined exception
///
/// * `dest` holds the return value if an invalid argument is given
/// * `kind` holds the kind of exception being raised
/// * `reason` holds the exception reason
/// * `trace` is optional, but holds the exception trace as an Erlang term
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Raise {
    pub dest: Register,
    pub kind: Register,
    pub reason: Register,
    pub trace: Option<Register>,
}
encoding_impl!(Raise, dest, kind, reason, trace);

/// Halts execution of the runtime with the given status/options
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Halt {
    pub status: Register,
    pub options: Register,
}
encoding_impl!(Halt, status, options);

/// Initializes a new binary builder
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BsInit {
    pub dest: Register,
}
encoding_impl!(BsInit, dest);

/// Finalizes an in-progress binary builder, producing the binary/bitstring value
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

/// Tests that `bin` is a binary/bitstring value, returning a boolean indicating if so,
/// and simultaneously starting a new match context and returning it
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BsMatchStart {
    pub is_err: Register,
    pub context: Register,
    pub bin: Register,
}
encoding_impl!(BsMatchStart, is_err, context, bin);

/// Attempts to match a value using the provided match context
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

/// Attempts to match an exact integer value using the provided match context, discarding the match result
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

/// Tests the size of the bitstring remaining in a match context, places a boolean in `dest` if the expected size is available
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BsTestTail {
    pub dest: Register,
    pub context: Register,
    pub size: usize,
}
encoding_impl!(BsTestTail, dest, context, size);

/// The first instruction in a function, providing metadata about the function
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FuncInfo {
    pub id: FunId,
    pub arity: Arity,
    pub frame_size: u8,
}
encoding_impl!(FuncInfo, id, arity, frame_size);

/// Puts the identity of the current process in `dest`
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
/// This can represent all variants of `spawn`, `spawn_link`, `spawn_monitor` that take a closure as an argument.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Spawn2 {
    pub dest: Register,
    pub fun: Register,
    pub opts: SpawnOpts,
}
encoding_impl!(Spawn2, dest, fun, opts);

/// Spawns a new process running `fun`, placing its pid in `dest`
///
/// This can represent all variants of `spawn`, `spawn_link`, `spawn_monitor` that take an MFA as an argument.
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
/// This can represent all variants of `spawn`, `spawn_link`, `spawn_monitor` that take an MFA as an argument.
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
            Self::Breq(_) => 7,
            Self::Call(_) => 8,
            Self::CallApply2(_) => 9,
            Self::CallApply3(_) => 10,
            Self::CallNative(_) => 11,
            Self::CallStatic(_) => 12,
            Self::CallIndirect(_) => 13,
            Self::Enter(_) => 14,
            Self::EnterApply2(_) => 15,
            Self::EnterApply3(_) => 16,
            Self::EnterNative(_) => 17,
            Self::EnterStatic(_) => 18,
            Self::EnterIndirect(_) => 19,
            Self::IsAtom(_) => 20,
            Self::IsBool(_) => 21,
            Self::IsNil(_) => 22,
            Self::IsTuple(_) => 23,
            Self::IsTupleFetchArity(_) => 24,
            Self::IsMap(_) => 25,
            Self::IsCons(_) => 26,
            Self::IsList(_) => 27,
            Self::IsInt(_) => 28,
            Self::IsFloat(_) => 29,
            Self::IsNumber(_) => 30,
            Self::IsPid(_) => 31,
            Self::IsRef(_) => 32,
            Self::IsPort(_) => 33,
            Self::IsBinary(_) => 34,
            Self::IsFunction(_) => 35,
            Self::LoadNil(_) => 36,
            Self::LoadBool(_) => 37,
            Self::LoadAtom(_) => 38,
            Self::LoadInt(_) => 39,
            Self::LoadBig(_) => 40,
            Self::LoadFloat(_) => 41,
            Self::LoadBinary(_) => 42,
            Self::LoadBitstring(_) => 43,
            Self::Not(_) => 44,
            Self::And(_) => 45,
            Self::AndAlso(_) => 46,
            Self::Or(_) => 47,
            Self::OrElse(_) => 48,
            Self::Xor(_) => 49,
            Self::Bnot(_) => 50,
            Self::Band(_) => 51,
            Self::Bor(_) => 52,
            Self::Bxor(_) => 53,
            Self::Bsl(_) => 54,
            Self::Bsr(_) => 55,
            Self::Div(_) => 56,
            Self::Rem(_) => 57,
            Self::Neg(_) => 58,
            Self::Add(_) => 59,
            Self::Sub(_) => 60,
            Self::Mul(_) => 61,
            Self::Divide(_) => 62,
            Self::ListAppend(_) => 63,
            Self::ListRemove(_) => 64,
            Self::Eq(_) => 65,
            Self::Neq(_) => 66,
            Self::Gt(_) => 67,
            Self::Gte(_) => 68,
            Self::Lt(_) => 69,
            Self::Lte(_) => 70,
            Self::Cons(_) => 71,
            Self::Split(_) => 72,
            Self::Head(_) => 73,
            Self::Tail(_) => 74,
            Self::Closure(_) => 75,
            Self::UnpackEnv(_) => 76,
            Self::Tuple(_) => 77,
            Self::TupleWithCapacity(_) => 78,
            Self::TupleArity(_) => 79,
            Self::GetElement(_) => 80,
            Self::SetElement(_) => 81,
            Self::SetElementMut(_) => 82,
            Self::Map(_) => 83,
            Self::MapPut(_) => 84,
            Self::MapPutMut(_) => 85,
            Self::MapUpdate(_) => 86,
            Self::MapUpdateMut(_) => 87,
            Self::MapExtendPut(_) => 88,
            Self::MapExtendUpdate(_) => 89,
            Self::MapTryGet(_) => 90,
            Self::Catch(_) => 91,
            Self::EndCatch(_) => 92,
            Self::LandingPad(_) => 93,
            Self::StackTrace(_) => 94,
            Self::Raise(_) => 95,
            Self::Send(_) => 96,
            Self::RecvPeek(_) => 97,
            Self::RecvNext(_) => 98,
            Self::RecvWait(_) => 99,
            Self::RecvTimeout(_) => 100,
            Self::RecvPop(_) => 101,
            Self::Yield(_) => 102,
            Self::GarbageCollect(_) => 103,
            Self::NormalExit(_) => 104,
            Self::ContinueExit(_) => 105,
            Self::Exit1(_) => 106,
            Self::Exit2(_) => 107,
            Self::Error1(_) => 108,
            Self::Throw1(_) => 109,
            Self::Halt(_) => 110,
            Self::BsInit(_) => 111,
            Self::BsPush(_) => 112,
            Self::BsFinish(_) => 113,
            Self::BsMatchStart(_) => 114,
            Self::BsMatch(_) => 115,
            Self::BsMatchSkip(_) => 116,
            Self::BsTestTail(_) => 117,
            Self::FuncInfo(_) => 118,
            Self::Identity(_) => 119,
            Self::Spawn2(_) => 120,
            Self::Spawn3(_) => 121,
            Self::Spawn3Indirect(_) => 122,
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
            7 => Ok(Self::Breq(Decode::decode(reader)?)),
            8 => Ok(Self::Call(Decode::decode(reader)?)),
            9 => Ok(Self::CallApply2(Decode::decode(reader)?)),
            10 => Ok(Self::CallApply3(Decode::decode(reader)?)),
            11 => Ok(Self::CallNative(Decode::decode(reader)?)),
            12 => Ok(Self::CallStatic(Decode::decode(reader)?)),
            13 => Ok(Self::CallIndirect(Decode::decode(reader)?)),
            14 => Ok(Self::Enter(Decode::decode(reader)?)),
            15 => Ok(Self::EnterApply2(Decode::decode(reader)?)),
            16 => Ok(Self::EnterApply3(Decode::decode(reader)?)),
            17 => Ok(Self::EnterNative(Decode::decode(reader)?)),
            18 => Ok(Self::EnterStatic(Decode::decode(reader)?)),
            19 => Ok(Self::EnterIndirect(Decode::decode(reader)?)),
            20 => Ok(Self::IsAtom(Decode::decode(reader)?)),
            21 => Ok(Self::IsBool(Decode::decode(reader)?)),
            22 => Ok(Self::IsNil(Decode::decode(reader)?)),
            23 => Ok(Self::IsTuple(Decode::decode(reader)?)),
            24 => Ok(Self::IsTupleFetchArity(Decode::decode(reader)?)),
            25 => Ok(Self::IsMap(Decode::decode(reader)?)),
            26 => Ok(Self::IsCons(Decode::decode(reader)?)),
            27 => Ok(Self::IsList(Decode::decode(reader)?)),
            28 => Ok(Self::IsInt(Decode::decode(reader)?)),
            29 => Ok(Self::IsFloat(Decode::decode(reader)?)),
            30 => Ok(Self::IsNumber(Decode::decode(reader)?)),
            31 => Ok(Self::IsPid(Decode::decode(reader)?)),
            32 => Ok(Self::IsRef(Decode::decode(reader)?)),
            33 => Ok(Self::IsPort(Decode::decode(reader)?)),
            34 => Ok(Self::IsBinary(Decode::decode(reader)?)),
            35 => Ok(Self::IsFunction(Decode::decode(reader)?)),
            36 => Ok(Self::LoadNil(Decode::decode(reader)?)),
            37 => Ok(Self::LoadBool(Decode::decode(reader)?)),
            38 => Ok(Self::LoadAtom(Decode::decode(reader)?)),
            39 => Ok(Self::LoadInt(Decode::decode(reader)?)),
            40 => Ok(Self::LoadBig(Decode::decode(reader)?)),
            41 => Ok(Self::LoadFloat(Decode::decode(reader)?)),
            42 => Ok(Self::LoadBinary(Decode::decode(reader)?)),
            43 => Ok(Self::LoadBitstring(Decode::decode(reader)?)),
            44 => Ok(Self::Not(Decode::decode(reader)?)),
            45 => Ok(Self::And(Decode::decode(reader)?)),
            46 => Ok(Self::AndAlso(Decode::decode(reader)?)),
            47 => Ok(Self::Or(Decode::decode(reader)?)),
            48 => Ok(Self::OrElse(Decode::decode(reader)?)),
            49 => Ok(Self::Xor(Decode::decode(reader)?)),
            50 => Ok(Self::Bnot(Decode::decode(reader)?)),
            51 => Ok(Self::Band(Decode::decode(reader)?)),
            52 => Ok(Self::Bor(Decode::decode(reader)?)),
            53 => Ok(Self::Bxor(Decode::decode(reader)?)),
            54 => Ok(Self::Bsl(Decode::decode(reader)?)),
            55 => Ok(Self::Bsr(Decode::decode(reader)?)),
            56 => Ok(Self::Div(Decode::decode(reader)?)),
            57 => Ok(Self::Rem(Decode::decode(reader)?)),
            58 => Ok(Self::Neg(Decode::decode(reader)?)),
            59 => Ok(Self::Add(Decode::decode(reader)?)),
            60 => Ok(Self::Sub(Decode::decode(reader)?)),
            61 => Ok(Self::Mul(Decode::decode(reader)?)),
            62 => Ok(Self::Divide(Decode::decode(reader)?)),
            63 => Ok(Self::ListAppend(Decode::decode(reader)?)),
            64 => Ok(Self::ListRemove(Decode::decode(reader)?)),
            65 => Ok(Self::Eq(Decode::decode(reader)?)),
            66 => Ok(Self::Neq(Decode::decode(reader)?)),
            67 => Ok(Self::Gt(Decode::decode(reader)?)),
            68 => Ok(Self::Gte(Decode::decode(reader)?)),
            69 => Ok(Self::Lt(Decode::decode(reader)?)),
            70 => Ok(Self::Lte(Decode::decode(reader)?)),
            71 => Ok(Self::Cons(Decode::decode(reader)?)),
            72 => Ok(Self::Split(Decode::decode(reader)?)),
            73 => Ok(Self::Head(Decode::decode(reader)?)),
            74 => Ok(Self::Tail(Decode::decode(reader)?)),
            75 => Ok(Self::Closure(Decode::decode(reader)?)),
            76 => Ok(Self::UnpackEnv(Decode::decode(reader)?)),
            77 => Ok(Self::Tuple(Decode::decode(reader)?)),
            78 => Ok(Self::TupleWithCapacity(Decode::decode(reader)?)),
            79 => Ok(Self::TupleArity(Decode::decode(reader)?)),
            80 => Ok(Self::GetElement(Decode::decode(reader)?)),
            81 => Ok(Self::SetElement(Decode::decode(reader)?)),
            82 => Ok(Self::SetElementMut(Decode::decode(reader)?)),
            83 => Ok(Self::Map(Decode::decode(reader)?)),
            84 => Ok(Self::MapPut(Decode::decode(reader)?)),
            85 => Ok(Self::MapPutMut(Decode::decode(reader)?)),
            86 => Ok(Self::MapUpdate(Decode::decode(reader)?)),
            87 => Ok(Self::MapUpdateMut(Decode::decode(reader)?)),
            88 => Ok(Self::MapExtendPut(Decode::decode(reader)?)),
            89 => Ok(Self::MapExtendUpdate(Decode::decode(reader)?)),
            90 => Ok(Self::MapTryGet(Decode::decode(reader)?)),
            91 => Ok(Self::Catch(Decode::decode(reader)?)),
            92 => Ok(Self::EndCatch(EndCatch)),
            93 => Ok(Self::LandingPad(Decode::decode(reader)?)),
            94 => Ok(Self::StackTrace(Decode::decode(reader)?)),
            95 => Ok(Self::Raise(Decode::decode(reader)?)),
            96 => Ok(Self::Send(Decode::decode(reader)?)),
            97 => Ok(Self::RecvPeek(Decode::decode(reader)?)),
            98 => Ok(Self::RecvNext(RecvNext)),
            99 => Ok(Self::RecvWait(Decode::decode(reader)?)),
            100 => Ok(Self::RecvTimeout(Decode::decode(reader)?)),
            101 => Ok(Self::RecvPop(RecvPop)),
            102 => Ok(Self::Yield(Yield)),
            103 => Ok(Self::GarbageCollect(Decode::decode(reader)?)),
            104 => Ok(Self::NormalExit(Decode::decode(reader)?)),
            105 => Ok(Self::ContinueExit(Decode::decode(reader)?)),
            106 => Ok(Self::Exit1(Decode::decode(reader)?)),
            107 => Ok(Self::Exit2(Decode::decode(reader)?)),
            108 => Ok(Self::Error1(Decode::decode(reader)?)),
            109 => Ok(Self::Throw1(Decode::decode(reader)?)),
            110 => Ok(Self::Halt(Decode::decode(reader)?)),
            111 => Ok(Self::BsInit(Decode::decode(reader)?)),
            112 => Ok(Self::BsPush(Decode::decode(reader)?)),
            113 => Ok(Self::BsFinish(Decode::decode(reader)?)),
            114 => Ok(Self::BsMatchStart(Decode::decode(reader)?)),
            115 => Ok(Self::BsMatch(Decode::decode(reader)?)),
            116 => Ok(Self::BsMatchSkip(Decode::decode(reader)?)),
            117 => Ok(Self::BsTestTail(Decode::decode(reader)?)),
            118 => Ok(Self::FuncInfo(Decode::decode(reader)?)),
            119 => Ok(Self::Identity(Decode::decode(reader)?)),
            120 => Ok(Self::Spawn2(Decode::decode(reader)?)),
            121 => Ok(Self::Spawn3(Decode::decode(reader)?)),
            122 => Ok(Self::Spawn3Indirect(Decode::decode(reader)?)),
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
            Self::Breq(op) => op.encode(writer),
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
        }
    }
}
