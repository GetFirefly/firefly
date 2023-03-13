use std::fmt;
use std::ops::{Deref, DerefMut};

use cranelift_entity::entity_impl;
use intrusive_collections::{intrusive_adapter, LinkedListLink, UnsafeRef};

use firefly_binary::BinaryEntrySpecifier;
use firefly_diagnostics::{Span, Spanned};
use firefly_syntax_base::{CallConv, Type};

use super::*;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Inst(u32);
entity_impl!(Inst, "inst");

#[derive(Clone, Spanned)]
pub struct InstNode {
    pub link: LinkedListLink,
    pub key: Inst,
    pub block: Block,
    #[span]
    pub data: Span<InstData>,
}
impl InstNode {
    pub fn new(key: Inst, block: Block, data: Span<InstData>) -> Self {
        Self {
            link: LinkedListLink::default(),
            key,
            block,
            data,
        }
    }
}
impl Deref for InstNode {
    type Target = Span<InstData>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
impl DerefMut for InstNode {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

intrusive_adapter!(pub InstAdapter = UnsafeRef<InstNode>: InstNode { link: LinkedListLink });

#[derive(Debug, Clone)]
pub enum InstData {
    BinaryOp(BinaryOp),
    BinaryOpImm(BinaryOpImm),
    UnaryOp(UnaryOp),
    UnaryOpImm(UnaryOpImm),
    UnaryOpConst(UnaryOpConst),
    Call(Call),
    CallIndirect(CallIndirect),
    Catch(Catch),
    MakeFun(MakeFun),
    Br(Br),
    CondBr(CondBr),
    Switch(Switch),
    Ret(Ret),
    PrimOp(PrimOp),
    PrimOpImm(PrimOpImm),
    IsType(IsType),
    BitsMatch(BitsMatch),
    BitsMatchSkip(BitsMatchSkip),
    BitsPush(BitsPush),
    SetElement(SetElement),
    SetElementImm(SetElementImm),
}
impl InstData {
    pub fn opcode(&self) -> Opcode {
        match self {
            Self::BinaryOp(BinaryOp { ref op, .. })
            | Self::BinaryOpImm(BinaryOpImm { ref op, .. })
            | Self::UnaryOp(UnaryOp { ref op, .. })
            | Self::UnaryOpImm(UnaryOpImm { ref op, .. })
            | Self::UnaryOpConst(UnaryOpConst { ref op, .. }) => *op,
            Self::Call(Call { ref op, .. }) => *op,
            Self::CallIndirect(CallIndirect { ref op, .. }) => *op,
            Self::Catch(Catch { ref op, .. }) => *op,
            Self::MakeFun(_) => Opcode::MakeFun,
            Self::Br(Br { ref op, .. }) => *op,
            Self::CondBr(_) => Opcode::CondBr,
            Self::Switch(Switch { ref op, .. }) => *op,
            Self::Ret(_) => Opcode::Ret,
            Self::PrimOp(PrimOp { ref op, .. }) | Self::PrimOpImm(PrimOpImm { ref op, .. }) => *op,
            Self::IsType(_) => Opcode::IsType,
            Self::BitsMatch(_) => Opcode::BitsMatch,
            Self::BitsMatchSkip(_) => Opcode::BitsMatchSkip,
            Self::BitsPush(_) => Opcode::BitsPush,
            Self::SetElement(SetElement { ref op, .. })
            | Self::SetElementImm(SetElementImm { ref op, .. }) => *op,
        }
    }

    pub fn arguments<'a>(&'a self, pool: &'a ValueListPool) -> &[Value] {
        match self {
            Self::BinaryOp(BinaryOp { ref args, .. }) => args.as_slice(),
            Self::BinaryOpImm(BinaryOpImm { ref arg, .. }) => core::slice::from_ref(arg),
            Self::UnaryOp(UnaryOp { ref arg, .. }) => core::slice::from_ref(arg),
            Self::UnaryOpImm(UnaryOpImm { .. }) => &[],
            Self::UnaryOpConst(UnaryOpConst { .. }) => &[],
            Self::Call(Call { ref args, .. }) => args.as_slice(pool),
            Self::CallIndirect(CallIndirect { ref args, .. }) => args.as_slice(pool),
            Self::Catch(_) => &[],
            Self::MakeFun(MakeFun { ref args, .. }) => args.as_slice(pool),
            Self::Br(Br { ref args, .. }) => args.as_slice(pool),
            Self::CondBr(CondBr { ref cond, .. }) => core::slice::from_ref(cond),
            Self::Switch(Switch { ref arg, .. }) => core::slice::from_ref(arg),
            Self::Ret(Ret { ref args, .. }) => args.as_slice(pool),
            Self::PrimOp(PrimOp { ref args, .. }) => args.as_slice(pool),
            Self::PrimOpImm(PrimOpImm { ref args, .. }) => args.as_slice(pool),
            Self::IsType(IsType { ref arg, .. }) => core::slice::from_ref(arg),
            Self::BitsMatch(BitsMatch { ref args, .. }) => args.as_slice(pool),
            Self::BitsMatchSkip(BitsMatchSkip { ref args, .. }) => args.as_slice(pool),
            Self::BitsPush(BitsPush { ref args, .. }) => args.as_slice(pool),
            Self::SetElement(SetElement { ref args, .. }) => args.as_slice(),
            Self::SetElementImm(SetElementImm { ref arg, .. }) => core::slice::from_ref(arg),
        }
    }

    pub fn arguments_mut<'a>(&'a mut self, pool: &'a mut ValueListPool) -> &mut [Value] {
        match self {
            Self::BinaryOp(BinaryOp { ref mut args, .. }) => args.as_mut_slice(),
            Self::BinaryOpImm(BinaryOpImm { ref mut arg, .. }) => core::slice::from_mut(arg),
            Self::UnaryOp(UnaryOp { ref mut arg, .. }) => core::slice::from_mut(arg),
            Self::UnaryOpImm(UnaryOpImm { .. }) => &mut [],
            Self::UnaryOpConst(UnaryOpConst { .. }) => &mut [],
            Self::Call(Call { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::CallIndirect(CallIndirect { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::Catch(_) => &mut [],
            Self::MakeFun(MakeFun { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::Br(Br { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::CondBr(CondBr { ref mut cond, .. }) => core::slice::from_mut(cond),
            Self::Switch(Switch { ref mut arg, .. }) => core::slice::from_mut(arg),
            Self::Ret(Ret { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::PrimOp(PrimOp { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::PrimOpImm(PrimOpImm { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::IsType(IsType { ref mut arg, .. }) => core::slice::from_mut(arg),
            Self::BitsMatch(BitsMatch { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::BitsMatchSkip(BitsMatchSkip { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::BitsPush(BitsPush { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::SetElement(SetElement { ref mut args, .. }) => args.as_mut_slice(),
            Self::SetElementImm(SetElementImm { ref mut arg, .. }) => core::slice::from_mut(arg),
        }
    }

    pub fn arguments_list<'a>(&'a mut self) -> Option<&mut ValueList> {
        match self {
            Self::CallIndirect(CallIndirect { ref mut args, .. }) => Some(args),
            Self::Br(Br { ref mut args, .. }) => Some(args),
            Self::PrimOp(PrimOp { ref mut args, .. }) => Some(args),
            Self::PrimOpImm(PrimOpImm { ref mut args, .. }) => Some(args),
            Self::BitsMatch(BitsMatch { ref mut args, .. }) => Some(args),
            Self::BitsMatchSkip(BitsMatchSkip { ref mut args, .. }) => Some(args),
            Self::BitsPush(BitsPush { ref mut args, .. }) => Some(args),
            _ => None,
        }
    }

    pub fn analyze_branch<'a>(&'a self, pool: &'a ValueListPool) -> BranchInfo<'a> {
        match self {
            Self::Br(ref b) if b.op == Opcode::Br => {
                BranchInfo::SingleDest(b.destination, b.args.as_slice(pool))
            }
            Self::Br(ref b) => BranchInfo::SingleDest(b.destination, &b.args.as_slice(pool)[1..]),
            Self::CondBr(CondBr {
                ref then_dest,
                ref else_dest,
                ..
            }) => BranchInfo::MultiDest(vec![
                JumpTable::new(then_dest.0, then_dest.1.as_slice(pool)),
                JumpTable::new(else_dest.0, else_dest.1.as_slice(pool)),
            ]),
            Self::Switch(Switch {
                ref arms,
                ref default,
                ..
            }) => {
                let mut targets = arms
                    .iter()
                    .map(|(_, b)| JumpTable::new(*b, &[]))
                    .collect::<Vec<_>>();
                targets.push(JumpTable::new(*default, &[]));
                BranchInfo::MultiDest(targets)
            }
            Self::Catch(Catch { dest, .. }) => BranchInfo::SingleDest(*dest, &[]),
            _ => BranchInfo::NotABranch,
        }
    }

    pub fn branch_destination(&self) -> Option<Block> {
        match self {
            Self::Br(ref b) => Some(b.destination),
            Self::Catch(Catch { dest, .. }) => Some(*dest),
            _ => None,
        }
    }

    pub fn analyze_call<'a>(&'a self, pool: &'a ValueListPool) -> CallInfo<'a> {
        match self {
            Self::Call(ref c) => CallInfo::Direct(c.callee, c.cc, c.args.as_slice(pool)),
            Self::CallIndirect(ref c) => CallInfo::Indirect(c.callee, c.cc, c.args.as_slice(pool)),
            _ => CallInfo::NotACall,
        }
    }
}

pub enum BranchInfo<'a> {
    NotABranch,
    SingleDest(Block, &'a [Value]),
    MultiDest(Vec<JumpTable<'a>>),
}

pub struct JumpTable<'a> {
    pub destination: Block,
    pub args: &'a [Value],
}
impl<'a> JumpTable<'a> {
    pub fn new(destination: Block, args: &'a [Value]) -> Self {
        Self { destination, args }
    }
}

pub enum CallInfo<'a> {
    NotACall,
    Direct(FuncRef, CallConv, &'a [Value]),
    Indirect(Value, CallConv, &'a [Value]),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Opcode {
    ImmInt,
    ImmFloat,
    ImmBool,
    ImmAtom,
    ImmNil,
    ImmNone,
    ImmNull,
    ConstBigInt,
    ConstBinary,
    IsNull,
    Cast,
    Trunc,
    Zext,
    Add,
    Sub,
    Mul,
    Div,
    Fdiv,
    Rem,
    Neg,
    Not,
    Bnot,
    IcmpEq,
    IcmpNeq,
    IcmpGt,
    IcmpGte,
    IcmpLt,
    IcmpLte,
    Eq,
    EqExact,
    Neq,
    NeqExact,
    Gt,
    Gte,
    Lt,
    Lte,
    And,
    Band,
    AndAlso,
    Or,
    Bor,
    OrElse,
    Xor,
    Bxor,
    Bsl,
    Bsr,
    Call,
    Enter,
    CallIndirect,
    EnterIndirect,
    CondBr,
    BrIf,
    BrUnless,
    Br,
    Switch,
    Ret,
    IsType,
    IsFunctionWithArity,
    IsTupleFetchArity,
    IsTaggedTuple,
    // List Operations
    Cons,
    Head,
    Tail,
    Split,
    ListConcat,
    ListSubtract,
    // Tuple Operations
    Tuple,
    GetElement,
    SetElement,
    SetElementMut,
    // Map Operations
    Map,
    MapPut,
    MapPutMut,
    MapUpdate,
    MapUpdateMut,
    MapExtendPut,
    MapExtendUpdate,
    MapTryGet,
    // Binary Operations
    BitsMatchStart,
    BitsMatch,
    BitsMatchSkip,
    BitsInit,
    BitsPush,
    BitsTestTail,
    BitsFinish,
    // Closures
    MakeFun,
    UnpackEnv,
    // Primops
    GarbageCollect,
    Send,
    RecvNext,
    RecvPeek,
    RecvPop,
    RecvWaitTimeout,
    NifStart,
    Yield,
    // Errors
    StartCatch,
    EndCatch,
    BuildStacktrace,
    Halt,
    Throw,
    Error,
    Exit1,
    Exit2,
    Raise,
}
impl Opcode {
    pub fn is_terminator(&self) -> bool {
        match self {
            Self::Br
            | Self::CondBr
            | Self::Switch
            | Self::Ret
            | Self::Halt
            | Self::Throw
            | Self::Error
            | Self::Exit1
            | Self::Enter
            | Self::EnterIndirect => true,
            _ => false,
        }
    }

    pub fn is_exception(&self) -> bool {
        match self {
            Self::Halt | Self::Throw | Self::Error | Self::Exit1 | Self::Raise => true,
            _ => false,
        }
    }

    pub fn num_fixed_args(&self) -> usize {
        match self {
            // Immediates/constants have none
            Self::ImmInt
            | Self::ImmFloat
            | Self::ImmBool
            | Self::ImmAtom
            | Self::ImmNil
            | Self::ImmNone
            | Self::ImmNull
            | Self::ConstBigInt
            | Self::ConstBinary => 0,
            // Binary ops always have two
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Fdiv
            | Self::Rem
            | Self::And
            | Self::Band
            | Self::AndAlso
            | Self::Or
            | Self::Bor
            | Self::OrElse
            | Self::Xor
            | Self::Bxor
            | Self::Bsl
            | Self::Bsr
            | Self::IcmpEq
            | Self::IcmpNeq
            | Self::IcmpGt
            | Self::IcmpGte
            | Self::IcmpLt
            | Self::IcmpLte
            | Self::Eq
            | Self::EqExact
            | Self::Neq
            | Self::NeqExact
            | Self::Gt
            | Self::Gte
            | Self::Lt
            | Self::Lte => 2,
            // Unary ops always have one
            Self::IsNull
            | Self::Cast
            | Self::Trunc
            | Self::Zext
            | Self::Neg
            | Self::Not
            | Self::Bnot
            | Self::IsType
            | Self::IsTupleFetchArity
            | Self::IsFunctionWithArity
            | Self::Head
            | Self::Tail
            | Self::Split => 1,
            // Tagged tuple checks take two arguments, the tuple and the tag
            Self::IsTaggedTuple => 2,
            // Tuple constructor takes a single argument, the arity
            Self::Tuple => 1,
            // Getting a tuple element takes the tuple and the index of the element
            Self::GetElement => 2,
            // Setting a tuple element takes two value arguments, the tuple, and the element, the
            // index is immediate
            Self::SetElement | Self::SetElementMut => 2,
            // Cons constructors/concat/subtract take two arguments, the head and tail
            // elements/lists
            Self::Cons | Self::ListConcat | Self::ListSubtract => 2,
            // Map constructor has no arguments
            Self::Map => 0,
            // Extending a map has at least 3 arguments, the base map and one key/value pair
            Self::MapExtendPut | Self::MapExtendUpdate => 3,
            // Getting a value from a map requires two arguments
            Self::MapTryGet => 2,
            // Map modifiers always have 3 arguments
            Self::MapPut | Self::MapUpdate | Self::MapPutMut | Self::MapUpdateMut => 3,
            // Creating a fun requires the current process and the callee; the environment is
            // variable-sized
            Self::MakeFun => 1,
            // Unpacking a closure environment requires the closure value
            Self::UnpackEnv => 1,
            // Calls are entirely variable
            Self::Call | Self::CallIndirect | Self::Enter | Self::EnterIndirect => 0,
            // Ifs have a single argument, the conditional
            Self::CondBr | Self::BrIf | Self::BrUnless => 1,
            // Unconditional branches have no fixed arguments
            Self::Br => 0,
            // Switches have a single argument, the input value
            Self::Switch => 1,
            // Returns require at least one argument
            Self::Ret => 1,
            // This receive intrinsic expects a timeout value as argument,
            Self::RecvWaitTimeout => 1,
            // These receive intrinsics manipulate "global" state, they don't take any arguments
            Self::RecvNext | Self::RecvPeek | Self::RecvPop => 0,
            // These primops expect either no arguments, an immediate or a value, so the number is
            // not fixed
            Self::GarbageCollect
            | Self::Halt
            | Self::BitsInit
            | Self::BitsMatchStart
            | Self::NifStart
            | Self::Yield => 0,
            // Except with these primitives which only require a value
            Self::Throw | Self::Error => 1,
            // Exit always has at least one argument, but may have two
            Self::Exit1 => 1,
            Self::Exit2 => 2,
            // Raise always has at least 2 arguments
            Self::Raise => 2,
            // Bitstring ops
            Self::BitsMatchSkip => 2,
            Self::BitsMatch | Self::BitsPush | Self::BitsFinish => 1,
            Self::BitsTestTail => 2,
            Self::StartCatch | Self::EndCatch | Self::BuildStacktrace => 0,
            Self::Send => 2,
        }
    }
}
impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::ImmInt => f.write_str("const.int"),
            Self::ImmFloat => f.write_str("const.float"),
            Self::ImmBool => f.write_str("const.bool"),
            Self::ImmAtom => f.write_str("const.atom"),
            Self::ImmNil => f.write_str("const.nil"),
            Self::ImmNone => f.write_str("const.none"),
            Self::ImmNull => f.write_str("null"),
            Self::ConstBigInt => f.write_str("const.bigint"),
            Self::ConstBinary => f.write_str("const.binary"),
            Self::IsNull => f.write_str("is_null"),
            Self::Cast => f.write_str("cast"),
            Self::Trunc => f.write_str("trunc"),
            Self::Zext => f.write_str("zext"),
            Self::CondBr => f.write_str("cond.br"),
            Self::Br => f.write_str("br"),
            Self::BrIf => f.write_str("br.if"),
            Self::BrUnless => f.write_str("br.unless"),
            Self::Switch => f.write_str("switch"),
            Self::Call => f.write_str("call"),
            Self::Enter => f.write_str("tail call"),
            Self::CallIndirect => f.write_str("call.indirect"),
            Self::EnterIndirect => f.write_str("tail call.indirect"),
            Self::Ret => f.write_str("ret"),
            Self::Add => f.write_str("add"),
            Self::Sub => f.write_str("sub"),
            Self::Mul => f.write_str("mul"),
            Self::Div => f.write_str("idiv"),
            Self::Fdiv => f.write_str("fdiv"),
            Self::Rem => f.write_str("rem"),
            Self::Neg => f.write_str("neg"),
            Self::And => f.write_str("and"),
            Self::Band => f.write_str("band"),
            Self::AndAlso => f.write_str("andalso"),
            Self::Or => f.write_str("or"),
            Self::Bor => f.write_str("bor"),
            Self::OrElse => f.write_str("orelse"),
            Self::Xor => f.write_str("xor"),
            Self::Bxor => f.write_str("bxor"),
            Self::Bsl => f.write_str("bsl"),
            Self::Bsr => f.write_str("bsr"),
            Self::IcmpEq => f.write_str("icmp.eq"),
            Self::IcmpNeq => f.write_str("icmp.neq"),
            Self::IcmpGt => f.write_str("icmp.gt"),
            Self::IcmpGte => f.write_str("icmp.gte"),
            Self::IcmpLt => f.write_str("icmp.lt"),
            Self::IcmpLte => f.write_str("icmp.lte"),
            Self::Eq => f.write_str("eq"),
            Self::EqExact => f.write_str("eq.exact"),
            Self::Neq => f.write_str("neq"),
            Self::NeqExact => f.write_str("neq.exact"),
            Self::Gt => f.write_str("gt"),
            Self::Gte => f.write_str("gte"),
            Self::Lt => f.write_str("lt"),
            Self::Lte => f.write_str("lte"),
            Self::Not => f.write_str("not"),
            Self::Bnot => f.write_str("bnot"),
            Self::IsType => f.write_str("is_type"),
            Self::IsTupleFetchArity => f.write_str("is_tuple_fetch_arity"),
            Self::IsFunctionWithArity => f.write_str("is_function"),
            Self::Cons => f.write_str("cons"),
            Self::Head => f.write_str("list.hd"),
            Self::Tail => f.write_str("list.tl"),
            Self::Split => f.write_str("list.split"),
            Self::ListConcat => f.write_str("list.concat"),
            Self::ListSubtract => f.write_str("list.subtract"),
            Self::Tuple => f.write_str("tuple"),
            Self::IsTaggedTuple => f.write_str("tuple.is_tagged"),
            Self::GetElement => f.write_str("tuple.get"),
            Self::SetElement => f.write_str("tuple.set"),
            Self::SetElementMut => f.write_str("tuple.set.mut"),
            Self::Map => f.write_str("map"),
            Self::MapPut => f.write_str("map.put"),
            Self::MapPutMut => f.write_str("map.put_mut"),
            Self::MapUpdate => f.write_str("map.update"),
            Self::MapUpdateMut => f.write_str("map.update_mut"),
            Self::MapExtendPut => f.write_str("map.extend.put"),
            Self::MapExtendUpdate => f.write_str("map.extend.update"),
            Self::MapTryGet => f.write_str("map.try_get"),
            Self::MakeFun => f.write_str("fun.make"),
            Self::UnpackEnv => f.write_str("fun.env.get"),
            Self::Send => f.write_str("send"),
            Self::RecvNext => f.write_str("recv.next"),
            Self::RecvPeek => f.write_str("recv.peek"),
            Self::RecvPop => f.write_str("recv.pop"),
            Self::RecvWaitTimeout => f.write_str("recv.wait_timeout"),
            Self::BitsMatchStart => f.write_str("bs.match.start"),
            Self::BitsMatch => f.write_str("bs.match"),
            Self::BitsMatchSkip => f.write_str("bs.match.skip"),
            Self::BitsInit => f.write_str("bs.init"),
            Self::BitsPush => f.write_str("bs.push"),
            Self::BitsTestTail => f.write_str("bs.test.tail"),
            Self::BitsFinish => f.write_str("bs.finish"),
            Self::StartCatch => f.write_str("catch.start"),
            Self::EndCatch => f.write_str("catch.end"),
            Self::BuildStacktrace => f.write_str("build_stacktrace"),
            Self::Halt => f.write_str("halt"),
            Self::Throw => f.write_str("throw"),
            Self::Error => f.write_str("error"),
            Self::Exit1 => f.write_str("exit1"),
            Self::Exit2 => f.write_str("exit2"),
            Self::Raise => f.write_str("raise"),
            Self::NifStart => f.write_str("nif.start"),
            Self::Yield => f.write_str("yield"),
            Self::GarbageCollect => f.write_str("garbage_collect"),
        }
    }
}
impl From<BinaryOpType> for Opcode {
    fn from(op: BinaryOpType) -> Self {
        match op {
            BinaryOpType::Add => Self::Add,
            BinaryOpType::Sub => Self::Sub,
            BinaryOpType::Mul => Self::Mul,
            BinaryOpType::Div => Self::Div,
            BinaryOpType::Fdiv => Self::Fdiv,
            BinaryOpType::Rem => Self::Rem,
            BinaryOpType::And => Self::And,
            BinaryOpType::Band => Self::Band,
            BinaryOpType::AndAlso => Self::AndAlso,
            BinaryOpType::Or => Self::Or,
            BinaryOpType::Bor => Self::Bor,
            BinaryOpType::OrElse => Self::OrElse,
            BinaryOpType::Xor => Self::Xor,
            BinaryOpType::Bxor => Self::Bxor,
            BinaryOpType::Bsl => Self::Bsl,
            BinaryOpType::Bsr => Self::Bsr,
            BinaryOpType::Eq => Self::Eq,
            BinaryOpType::EqExact => Self::EqExact,
            BinaryOpType::Neq => Self::Neq,
            BinaryOpType::NeqExact => Self::NeqExact,
            BinaryOpType::Gt => Self::Gt,
            BinaryOpType::Gte => Self::Gte,
            BinaryOpType::Lt => Self::Lt,
            BinaryOpType::Lte => Self::Lte,
            BinaryOpType::ListConcat => Self::ListConcat,
            BinaryOpType::ListSubtract => Self::ListSubtract,
        }
    }
}
impl From<UnaryOpType> for Opcode {
    fn from(op: UnaryOpType) -> Self {
        match op {
            UnaryOpType::Not => Self::Not,
            UnaryOpType::Bnot => Self::Bnot,
            UnaryOpType::Neg => Self::Neg,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BinaryOp {
    pub op: Opcode,
    pub args: [Value; 2],
}

#[derive(Debug, Clone)]
pub struct BinaryOpImm {
    pub op: Opcode,
    pub arg: Value,
    pub imm: Immediate,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinaryOpType {
    Add,
    Sub,
    Mul,
    Div,
    Fdiv,
    Rem,
    And,
    Band,
    AndAlso,
    Or,
    Bor,
    OrElse,
    Xor,
    Bxor,
    Bsl,
    Bsr,
    Eq,
    EqExact,
    Neq,
    NeqExact,
    Gt,
    Gte,
    Lt,
    Lte,
    ListConcat,
    ListSubtract,
}

#[derive(Debug, Clone)]
pub struct UnaryOp {
    pub op: Opcode,
    pub arg: Value,
}

#[derive(Debug, Clone)]
pub struct UnaryOpImm {
    pub op: Opcode,
    pub imm: Immediate,
}

#[derive(Debug, Clone)]
pub struct UnaryOpConst {
    pub op: Opcode,
    pub imm: Constant,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UnaryOpType {
    Not,
    Bnot,
    Neg,
}

#[derive(Debug, Clone)]
pub struct Call {
    pub op: Opcode,
    pub cc: CallConv,
    pub callee: FuncRef,
    pub args: ValueList,
}

#[derive(Debug, Clone)]
pub struct CallIndirect {
    pub op: Opcode,
    pub cc: CallConv,
    pub callee: Value,
    pub args: ValueList,
}

/// Branch
#[derive(Debug, Clone)]
pub struct Br {
    pub op: Opcode,
    pub destination: Block,
    pub args: ValueList,
}

/// Conditional Branch
#[derive(Debug, Clone)]
pub struct CondBr {
    pub cond: Value,
    pub then_dest: (Block, ValueList),
    pub else_dest: (Block, ValueList),
}

/// Start of catch region
#[derive(Debug, Clone)]
pub struct Catch {
    pub op: Opcode,
    pub dest: Block,
}

/// Switch-case
#[derive(Debug, Clone)]
pub struct Switch {
    pub op: Opcode,
    pub arg: Value,
    pub arms: Vec<(u32, Block)>,
    pub default: Block,
}

/// Return
#[derive(Debug, Clone)]
pub struct Ret {
    pub op: Opcode,
    pub args: ValueList,
}

/// A primop that takes a variable number of terms
#[derive(Debug, Clone)]
pub struct PrimOp {
    pub op: Opcode,
    pub args: ValueList,
}

/// A primop that takes an immediate for its first argument, followed by a variable number of terms
#[derive(Debug, Clone)]
pub struct PrimOpImm {
    pub op: Opcode,
    pub imm: Immediate,
    pub args: ValueList,
}

#[derive(Debug, Clone)]
pub struct IsType {
    pub arg: Value,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct MakeFun {
    pub callee: FuncRef,
    /// This argument list always has at least one value, the current process;
    /// the remainder are the variable-sized closure env values
    pub args: ValueList,
}

#[derive(Debug, Clone)]
pub struct BitsMatch {
    pub spec: BinaryEntrySpecifier,
    pub args: ValueList,
}

#[derive(Debug, Clone)]
pub struct BitsMatchSkip {
    pub spec: BinaryEntrySpecifier,
    pub args: ValueList,
    pub value: Immediate,
}

#[derive(Debug, Clone)]
pub struct BitsPush {
    pub spec: BinaryEntrySpecifier,
    pub args: ValueList,
}

#[derive(Debug, Clone)]
pub struct SetElement {
    pub op: Opcode,
    pub index: Immediate,
    pub args: [Value; 2],
}

/// SetElement, but with an immediate index and value
#[derive(Debug, Clone)]
pub struct SetElementImm {
    pub op: Opcode,
    pub arg: Value,
    pub index: Immediate,
    pub value: Immediate,
}
