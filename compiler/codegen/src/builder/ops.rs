mod builder;
pub(super) mod builders;

use std::fmt;

use libeir_intern::Ident;
use libeir_ir as ir;
use libeir_ir::FunctionIdent;

use liblumen_mlir::ir::LocationRef;

pub use self::builder::OpBuilder;

use crate::Result;

use super::block::Block;
use super::ffi::{self, Type};
use super::value::Value;
use super::ScopedFunctionBuilder;

/// Represents the different types of operations that can be present in an EIR block
#[derive(Debug, Clone)]
pub enum OpKind {
    Return(Return),
    Throw(Throw),
    Unreachable(LocationRef),
    Branch(Br),
    Call(Call),
    If(If),
    IsType(IsType),
    Match(Match),
    BinaryPush(BinaryPush),
    MapPut(MapPuts),
    BinOp(BinaryOperator),
    LogicOp(LogicalOperator),
    #[allow(dead_code)]
    Constant(Constant),
    FunctionRef(FunctionRef),
    Tuple(Tuple),
    Cons(Cons),
    Map(Map),
    TraceCapture(TraceCapture),
    TraceConstruct(TraceConstruct),
    Intrinsic(Intrinsic),
}

#[derive(Debug, Clone)]
pub struct Br {
    pub loc: LocationRef,
    pub dest: Branch,
}

#[derive(Debug, Clone)]
pub struct IsType {
    pub loc: LocationRef,
    pub value: Value,
    pub expected: Type,
}

#[derive(Debug, Clone)]
pub struct Constant {
    pub loc: LocationRef,
    pub constant: ir::Const,
}

#[derive(Debug, Clone)]
pub struct Tuple {
    pub loc: LocationRef,
    pub elements: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct Cons {
    pub loc: LocationRef,
    pub head: Value,
    pub tail: Value,
}

#[derive(Debug, Clone)]
pub struct Map {
    pub loc: LocationRef,
    pub elements: Vec<(Value, Value)>,
}

#[derive(Debug, Clone, Copy)]
pub struct Return {
    pub loc: LocationRef,
    pub value: Option<Value>,
}

#[derive(Debug, Clone, Copy)]
pub struct BinaryOperator {
    pub loc: LocationRef,
    pub kind: ir::BinOp,
    pub lhs: Value,
    pub rhs: Value,
}

#[derive(Debug, Clone, Copy)]
pub struct LogicalOperator {
    pub loc: LocationRef,
    pub kind: ir::LogicOp,
    pub lhs: Value,
    pub rhs: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct ClosureInfo {
    pub ident: FunctionIdent,
    pub index: u32,
    pub old_unique: u32,
    pub unique: [u8; 16],
}

#[derive(Debug, Clone)]
pub struct FunctionRef {
    pub loc: LocationRef,
    pub callee: Callee,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Callee {
    Static(FunctionIdent),
    LocalDynamic {
        module: Ident,
        function: Value,
        arity: usize,
    },
    GlobalDynamic {
        module: Value,
        function: Value,
        arity: usize,
    },
    ClosureDynamic(Value),
}
impl Callee {
    pub fn new<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        callee_value: ir::Value,
    ) -> Result<Self> {
        use libeir_ir::ValueKind;

        let callee_kind = builder.value_kind(callee_value);
        builder.debug(&format!(
            "callee value is {:?} ({:?})",
            callee_value, callee_kind
        ));

        // Handle calls to closures
        if let ir::ValueKind::Argument(_, _) = callee_kind {
            return Ok(Self::ClosureDynamic(builder.build_value(callee_value)?));
        }

        let op = builder.get_primop(callee_value);
        debug_assert_eq!(ir::PrimOpKind::CaptureFunction, *builder.primop_kind(op));
        let reads = builder.primop_reads(op);
        let num_reads = reads.len();
        // Figure out if this is a statically known function
        assert_eq!(3, num_reads, "expected 3 arguments to capture function op");

        // Resolve arity first, since we should always know arity
        let arity = builder.constant_int(builder.value_const(reads[2]));

        let m = reads[0];
        let f = reads[1];
        let mk = builder.value_kind(m);
        let callee = if let ValueKind::Const(mc) = mk {
            let module = builder.constant_atom(mc);
            let fk = builder.value_kind(f);
            if let ValueKind::Const(fc) = fk {
                let function = builder.constant_atom(fc);
                Self::Static(FunctionIdent {
                    module: Ident::with_empty_span(module),
                    name: Ident::with_empty_span(function),
                    arity: arity as usize,
                })
            } else {
                let function = builder.build_value(f)?;
                // Locally dynamic
                Self::LocalDynamic {
                    module: Ident::with_empty_span(module),
                    function,
                    arity: arity as usize,
                }
            }
        } else {
            let module = builder.build_value(m)?;
            let function = builder.build_value(f)?;
            // Globally dynamic
            Self::GlobalDynamic {
                module,
                function,
                arity: arity as usize,
            }
        };
        Ok(callee)
    }
}
impl fmt::Display for Callee {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Static(ref ident) => write!(f, "{}", ident),
            Self::LocalDynamic {
                module,
                function,
                arity,
            } => write!(f, "{}:{:?}/{}", module, function, arity),
            Self::GlobalDynamic {
                module,
                function,
                arity,
            } => write!(f, "{:?}:{:?}/{}", module, function, arity),
            Self::ClosureDynamic(value) => write!(f, "<closure::{:?}>", value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Call {
    pub loc: LocationRef,
    pub callee: Callee,
    pub args: Vec<Value>,
    pub is_tail: bool,
    pub ok: CallSuccess,
    pub err: CallError,
}

#[derive(Debug, Clone)]
pub enum CallSuccess {
    Return,
    Branch(Branch),
}

#[derive(Debug, Clone)]
pub enum CallError {
    Throws,
    Catch(Branch),
}

#[derive(Debug, Clone)]
pub struct If {
    pub loc: LocationRef,
    pub cond: Value,
    pub yes: Branch,
    pub no: Branch,
    pub otherwise: Option<Branch>,
}

#[derive(Debug, Clone)]
pub struct Branch {
    pub block: Block,
    pub args: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct Match {
    pub loc: LocationRef,
    pub selector: Value,
    pub branches: Vec<Pattern>,
    pub reads: Vec<ir::Value>,
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub loc: LocationRef,
    pub kind: ir::MatchKind,
    pub block: Block,
    pub args: Vec<ir::Value>,
}

#[derive(Debug, Clone)]
pub struct TraceCapture {
    pub loc: LocationRef,
    pub dest: Branch,
}

#[derive(Debug, Clone)]
pub struct TraceConstruct {
    pub loc: LocationRef,
    pub capture: Value,
}

#[derive(Debug, Clone)]
pub struct Intrinsic {
    pub loc: LocationRef,
    pub name: libeir_intern::Symbol,
    pub args: Vec<ir::Value>,
}

#[derive(Debug, Clone)]
pub struct BinaryPush {
    pub loc: LocationRef,
    pub ok: Block,
    pub err: Block,
    pub head: Value,
    pub tail: Value,
    pub size: Option<Value>,
    pub spec: ir::BinaryEntrySpecifier,
}

#[derive(Debug, Clone)]
pub struct MapPuts {
    pub loc: LocationRef,
    pub ok: Block,
    pub err: Block,
    pub map: Value,
    pub puts: Vec<MapPut>,
}

#[derive(Debug, Clone)]
pub struct MapPut {
    pub action: ffi::MapActionType,
    pub key: Value,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub struct Throw {
    pub loc: LocationRef,
    pub kind: Value,
    pub reason: Value,
    pub trace: Value,
}
