use std::fmt;
use std::ops::{Deref, DerefMut};

use cranelift_entity::entity_impl;
use intrusive_collections::{intrusive_adapter, LinkedListLink, UnsafeRef};
use liblumen_binary::BinaryEntrySpecifier;
use liblumen_diagnostics::Spanned;

use super::*;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Inst(u32);
entity_impl!(Inst, "inst");

#[derive(Clone)]
pub struct InstNode {
    pub link: LinkedListLink,
    pub key: Inst,
    pub block: Block,
    pub data: Spanned<InstData>,
}
impl InstNode {
    pub fn new(key: Inst, block: Block, data: Spanned<InstData>) -> Self {
        Self {
            link: LinkedListLink::default(),
            key,
            block,
            data,
        }
    }
}
impl Deref for InstNode {
    type Target = Spanned<InstData>;

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
    BinaryOpConst(BinaryOpConst),
    UnaryOp(UnaryOp),
    UnaryOpImm(UnaryOpImm),
    UnaryOpConst(UnaryOpConst),
    Call(Call),
    CallIndirect(CallIndirect),
    Br(Br),
    Ret(Ret),
    RetImm(RetImm),
    PrimOp(PrimOp),
    PrimOpImm(PrimOpImm),
    IsType(IsType),
    BitsMatch(BitsMatch),
    BitsPush(BitsPush),
    SetElement(SetElement),
    SetElementImm(SetElementImm),
    SetElementConst(SetElementConst),
    MapUpdate(MapUpdate),
}
impl InstData {
    pub fn opcode(&self) -> Opcode {
        match self {
            Self::BinaryOp(BinaryOp { ref op, .. })
            | Self::BinaryOpImm(BinaryOpImm { ref op, .. })
            | Self::BinaryOpConst(BinaryOpConst { ref op, .. })
            | Self::UnaryOp(UnaryOp { ref op, .. })
            | Self::UnaryOpImm(UnaryOpImm { ref op, .. })
            | Self::UnaryOpConst(UnaryOpConst { ref op, .. }) => *op,
            Self::Call(_) => Opcode::Call,
            Self::CallIndirect(_) => Opcode::CallIndirect,
            Self::Br(Br { ref op, .. }) => *op,
            Self::Ret(_) | Self::RetImm(_) => Opcode::Ret,
            Self::PrimOp(PrimOp { ref op, .. }) | Self::PrimOpImm(PrimOpImm { ref op, .. }) => *op,
            Self::IsType(_) => Opcode::IsType,
            Self::BitsMatch(_) => Opcode::BitsMatch,
            Self::BitsPush(_) => Opcode::BitsPush,
            Self::SetElement(_) | Self::SetElementImm(_) | Self::SetElementConst(_) => {
                Opcode::SetElement
            }
            Self::MapUpdate(MapUpdate { ref op, .. }) => *op,
        }
    }

    pub fn arguments<'a>(&'a self, pool: &'a ValueListPool) -> &[Value] {
        match self {
            Self::BinaryOp(BinaryOp { ref args, .. }) => args.as_slice(),
            Self::BinaryOpImm(BinaryOpImm { ref arg, .. }) => core::slice::from_ref(arg),
            Self::BinaryOpConst(BinaryOpConst { ref arg, .. }) => core::slice::from_ref(arg),
            Self::UnaryOp(UnaryOp { ref arg, .. }) => core::slice::from_ref(arg),
            Self::UnaryOpImm(UnaryOpImm { .. }) => &[],
            Self::UnaryOpConst(UnaryOpConst { .. }) => &[],
            Self::Call(Call { ref args, .. }) => args.as_slice(pool),
            Self::CallIndirect(CallIndirect { ref args, .. }) => args.as_slice(pool),
            Self::Br(Br { ref args, .. }) => args.as_slice(pool),
            Self::Ret(Ret { ref args, .. }) => args.as_slice(),
            Self::RetImm(RetImm { ref arg, .. }) => core::slice::from_ref(arg),
            Self::PrimOp(PrimOp { ref args, .. }) => args.as_slice(pool),
            Self::PrimOpImm(PrimOpImm { ref args, .. }) => args.as_slice(pool),
            Self::IsType(IsType { ref arg, .. }) => core::slice::from_ref(arg),
            Self::BitsMatch(BitsMatch { ref args, .. }) => args.as_slice(pool),
            Self::BitsPush(BitsPush { ref args, .. }) => args.as_slice(pool),
            Self::SetElement(SetElement { ref args, .. }) => args.as_slice(),
            Self::SetElementImm(SetElementImm { ref arg, .. }) => core::slice::from_ref(arg),
            Self::SetElementConst(SetElementConst { ref arg, .. }) => core::slice::from_ref(arg),
            Self::MapUpdate(MapUpdate { ref args, .. }) => args.as_slice(),
        }
    }

    pub fn arguments_mut<'a>(&'a mut self, pool: &'a mut ValueListPool) -> &mut [Value] {
        match self {
            Self::BinaryOp(BinaryOp { ref mut args, .. }) => args.as_mut_slice(),
            Self::BinaryOpImm(BinaryOpImm { ref mut arg, .. }) => core::slice::from_mut(arg),
            Self::BinaryOpConst(BinaryOpConst { ref mut arg, .. }) => core::slice::from_mut(arg),
            Self::UnaryOp(UnaryOp { ref mut arg, .. }) => core::slice::from_mut(arg),
            Self::UnaryOpImm(UnaryOpImm { .. }) => &mut [],
            Self::UnaryOpConst(UnaryOpConst { .. }) => &mut [],
            Self::Call(Call { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::CallIndirect(CallIndirect { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::Br(Br { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::Ret(Ret { ref mut args, .. }) => args.as_mut_slice(),
            Self::RetImm(RetImm { ref mut arg, .. }) => core::slice::from_mut(arg),
            Self::PrimOp(PrimOp { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::PrimOpImm(PrimOpImm { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::IsType(IsType { ref mut arg, .. }) => core::slice::from_mut(arg),
            Self::BitsMatch(BitsMatch { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::BitsPush(BitsPush { ref mut args, .. }) => args.as_mut_slice(pool),
            Self::SetElement(SetElement { ref mut args, .. }) => args.as_mut_slice(),
            Self::SetElementImm(SetElementImm { ref mut arg, .. }) => core::slice::from_mut(arg),
            Self::SetElementConst(SetElementConst { ref mut arg, .. }) => {
                core::slice::from_mut(arg)
            }
            Self::MapUpdate(MapUpdate { ref mut args, .. }) => args.as_mut_slice(),
        }
    }

    pub fn arguments_list<'a>(&'a mut self) -> Option<&mut ValueList> {
        match self {
            Self::CallIndirect(CallIndirect { ref mut args, .. }) => Some(args),
            Self::Br(Br { ref mut args, .. }) => Some(args),
            Self::PrimOp(PrimOp { ref mut args, .. }) => Some(args),
            Self::PrimOpImm(PrimOpImm { ref mut args, .. }) => Some(args),
            Self::BitsMatch(BitsMatch { ref mut args, .. }) => Some(args),
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
            _ => BranchInfo::NotABranch,
        }
    }

    pub fn branch_destination(&self) -> Option<Block> {
        match self {
            Self::Br(ref b) => Some(b.destination),
            _ => None,
        }
    }

    pub fn analyze_call<'a>(&'a self, pool: &'a ValueListPool) -> CallInfo<'a> {
        match self {
            Self::Call(ref c) => CallInfo::Direct(c.callee, c.args.as_slice(pool)),
            Self::CallIndirect(ref c) => CallInfo::Indirect(c.callee, c.args.as_slice(pool)),
            _ => CallInfo::NotACall,
        }
    }
}

pub enum BranchInfo<'a> {
    NotABranch,
    SingleDest(Block, &'a [Value]),
}

pub enum CallInfo<'a> {
    NotACall,
    Direct(FuncRef, &'a [Value]),
    Indirect(Value, &'a [Value]),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Opcode {
    ImmInt,
    ImmFloat,
    ImmBool,
    ImmAtom,
    ImmNil,
    ImmNone,
    ConstBigInt,
    ConstBinary,
    ConstTuple,
    ConstList,
    ConstMap,
    Add,
    Sub,
    Mul,
    Div,
    Fdiv,
    Rem,
    Neg,
    Not,
    Bnot,
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
    CallIndirect,
    BrIf,
    BrUnless,
    Br,
    Ret,
    IsType,
    IsTaggedTuple,
    // List Operations
    Cons,
    Head,
    Tail,
    ListConcat,
    ListSubtract,
    // Tuple Operations
    Tuple,
    GetElement,
    SetElement,
    // Map Operations
    Map,
    MapPut,
    MapUpdate,
    // Binary Operations
    BitsMatch,
    BitsInitWritable,
    BitsPush,
    BitsCloseWritable,
    // Closures
    MakeFun,
    CaptureFun,
    // Primops
    MatchFail,
    RecvStart,
    RecvNext,
    RecvPeek,
    RecvPop,
    RecvWait,
    RecvDone,
    NifStart,
    // Errors
    Raise,
    BuildStacktrace,
    ExceptionClass,
    ExceptionReason,
    ExceptionTrace,
}
impl Opcode {
    pub fn is_terminator(&self) -> bool {
        match self {
            Self::Br | Self::Ret | Self::Raise | Self::MatchFail => true,
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
            | Self::ConstBigInt
            | Self::ConstBinary
            | Self::ConstTuple
            | Self::ConstList
            | Self::ConstMap => 0,
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
            | Self::Eq
            | Self::EqExact
            | Self::Neq
            | Self::NeqExact
            | Self::Gt
            | Self::Gte
            | Self::Lt
            | Self::Lte => 2,
            // Unary ops always have one
            Self::Neg | Self::Not | Self::Bnot | Self::IsType | Self::Head | Self::Tail => 1,
            // Tagged tuple checks take two arguments, the tuple and the tag
            Self::IsTaggedTuple => 2,
            // Tuple constructor takes a single argument, the arity
            Self::Tuple => 1,
            // Getting a tuple element takes the tuple and the index of the element
            Self::GetElement => 2,
            // Setting a tuple element takes three arguments, the tuple, the index, and the element
            Self::SetElement => 3,
            // Cons constructors/concat/subtract take two arguments, the head and tail elements/lists
            Self::Cons | Self::ListConcat | Self::ListSubtract => 2,
            // Creating a map has no arguments
            Self::Map => 0,
            // Inserting/updating a map takes 3 arguments, map/key/value
            Self::MapPut | Self::MapUpdate => 3,
            // Creating a fun only requires the callee, the environment is variable-sized
            Self::MakeFun => 1,
            // Capturing a fun requires the module, function, and arity
            Self::CaptureFun => 3,
            // Calls are entirely variable
            Self::Call | Self::CallIndirect => 0,
            // Ifs have a single argument, the conditional
            Self::BrIf | Self::BrUnless => 1,
            // Unconditional branches have no fixed arguments
            Self::Br => 0,
            // Returns require at least one argument, but in general two are present, the second being the exception flag
            Self::Ret => 1,
            // The following primops expect no arguments
            Self::BuildStacktrace => 0,
            // This receive intrinsic expects a timeout value as argument,
            Self::RecvStart => 1,
            // These receive intrinsics expect the receive context as argument
            Self::RecvNext | Self::RecvPeek | Self::RecvPop | Self::RecvWait | Self::RecvDone => 1,
            // These exception primops expect the exception value
            Self::ExceptionClass | Self::ExceptionReason | Self::ExceptionTrace => 1,
            // These primops expect either an immediate or a value, so the number is not fixed
            Self::MatchFail | Self::BitsInitWritable | Self::NifStart => 0,
            // Raising errors requires the class, the error value, and the stacktrace
            Self::Raise => 3,
            // Binary ops
            Self::BitsMatch | Self::BitsPush | Self::BitsCloseWritable => 1,
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
            Self::ConstBigInt => f.write_str("const.bigint"),
            Self::ConstBinary => f.write_str("const.binary"),
            Self::ConstTuple => f.write_str("const.tuple"),
            Self::ConstList => f.write_str("const.list"),
            Self::ConstMap => f.write_str("const.map"),
            Self::Br => f.write_str("br"),
            Self::BrIf => f.write_str("br.if"),
            Self::BrUnless => f.write_str("br.unless"),
            Self::Call => f.write_str("call"),
            Self::CallIndirect => f.write_str("call.indirect"),
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
            Self::Cons => f.write_str("cons"),
            Self::Head => f.write_str("list.hd"),
            Self::Tail => f.write_str("list.tl"),
            Self::ListConcat => f.write_str("list.concat"),
            Self::ListSubtract => f.write_str("list.subtract"),
            Self::Tuple => f.write_str("tuple"),
            Self::Map => f.write_str("map"),
            Self::MapPut => f.write_str("map.put"),
            Self::MapUpdate => f.write_str("map.update"),
            Self::IsTaggedTuple => f.write_str("tuple.is_tagged"),
            Self::GetElement => f.write_str("tuple.get"),
            Self::SetElement => f.write_str("tuple.set"),
            Self::MakeFun => f.write_str("fun.make"),
            Self::CaptureFun => f.write_str("fun.capture"),
            Self::MatchFail => f.write_str("match_fail"),
            Self::RecvStart => f.write_str("recv.start"),
            Self::RecvNext => f.write_str("recv.next"),
            Self::RecvPeek => f.write_str("recv.peek"),
            Self::RecvPop => f.write_str("recv.pop"),
            Self::RecvWait => f.write_str("recv.wait"),
            Self::RecvDone => f.write_str("recv.done"),
            Self::BitsMatch => f.write_str("bs.match"),
            Self::BitsInitWritable => f.write_str("bs.init"),
            Self::BitsPush => f.write_str("bs.push"),
            Self::BitsCloseWritable => f.write_str("bs.finish"),
            Self::Raise => f.write_str("raise"),
            Self::NifStart => f.write_str("nif.start"),
            Self::BuildStacktrace => f.write_str("stacktrace.build"),
            Self::ExceptionClass => f.write_str("exception.class"),
            Self::ExceptionReason => f.write_str("exception.reason"),
            Self::ExceptionTrace => f.write_str("exception.trace"),
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

#[derive(Debug, Clone)]
pub struct BinaryOpConst {
    pub op: Opcode,
    pub arg: Value,
    pub imm: Constant,
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
    pub callee: FuncRef,
    pub args: ValueList,
}

#[derive(Debug, Clone)]
pub struct CallIndirect {
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

/// Return
#[derive(Debug, Clone)]
pub struct Ret {
    pub op: Opcode,
    pub args: [Value; 2],
}

/// Return w/ immediate exception flag
#[derive(Debug, Clone)]
pub struct RetImm {
    pub op: Opcode,
    pub arg: Value,
    pub imm: Immediate,
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
pub struct BitsMatch {
    pub spec: BinaryEntrySpecifier,
    pub args: ValueList,
}

#[derive(Debug, Clone)]
pub struct BitsPush {
    pub spec: BinaryEntrySpecifier,
    pub args: ValueList,
}

#[derive(Debug, Clone)]
pub struct SetElement {
    pub op: Opcode,
    pub args: [Value; 3],
}

/// SetElement, but with an immediate index and value
#[derive(Debug, Clone)]
pub struct SetElementImm {
    pub op: Opcode,
    pub arg: Value,
    pub index: Immediate,
    pub value: Immediate,
}

/// SetElement, but with an immediate index and constant value
#[derive(Debug, Clone)]
pub struct SetElementConst {
    pub op: Opcode,
    pub arg: Value,
    pub index: Immediate,
    pub value: Constant,
}

/// Used for both map insertions and updates
#[derive(Debug, Clone)]
pub struct MapUpdate {
    pub op: Opcode,
    pub args: [Value; 3],
}
