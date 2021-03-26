use std::convert::From;

use liblumen_compiler_macros::foreign_struct;
use liblumen_llvm as llvm;
use liblumen_mlir::ir::*;
use liblumen_mlir::{ContextRef, ModuleRef};
use liblumen_target::Endianness;

use crate::builder::function::Param;

#[foreign_struct]
pub struct ModuleBuilder;

#[repr(C)]
pub struct ModuleBuilderResult {
    pub module: ModuleRef,
    pub success: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SourceLocation {
    pub filename: *const libc::c_char,
    pub line: u32,
    pub column: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Span {
    start: u32,
    end: u32,
}
impl Span {
    #[inline]
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }
}
impl Default for Span {
    #[inline]
    fn default() -> Span {
        Self { start: 0, end: 0 }
    }
}
impl From<liblumen_util::diagnostics::SourceSpan> for Span {
    fn from(span: liblumen_util::diagnostics::SourceSpan) -> Span {
        Self {
            start: span.start().to_usize() as u32,
            end: span.end().to_usize() as u32,
        }
    }
}

/// Returned when creating a function in a module
///
/// Creating a function also provides us with an entry
/// block which matches the parameter list of the function,
/// rather than make an extra FFI call to get the entry block,
/// we just return both the function reference and the entry block
/// reference in one wrapper struct. Ideally, we'd use a tuple for this.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FunctionDeclResult {
    pub function: FunctionOpRef,
    pub entry_block: BlockRef,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Closure {
    pub loc: LocationRef,
    pub module: AttributeRef,
    pub name: *const libc::c_char,
    pub arity: u8,
    pub index: u32,
    pub old_unique: u32,
    pub unique: [u8; 16],
    pub env: *const ValueRef,
    pub env_len: libc::c_uint,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(C)]
pub enum BinarySpecifier {
    Integer {
        signed: bool,
        endianness: Endianness,
        unit: i64,
    },
    Float {
        endianness: Endianness,
        unit: i64,
    },
    Bytes {
        unit: i64,
    },
    Bits {
        unit: i64,
    },
    Utf8,
    Utf16 {
        endianness: Endianness,
    },
    Utf32 {
        endianness: Endianness,
    },
}
impl From<&libeir_ir::BinaryEntrySpecifier> for BinarySpecifier {
    fn from(s: &libeir_ir::BinaryEntrySpecifier) -> Self {
        use libeir_ir::BinaryEntrySpecifier::*;
        match *s {
            Integer {
                signed,
                endianness,
                unit,
            } => BinarySpecifier::Integer {
                signed,
                endianness: eir_to_target_endianness(endianness),
                unit,
            },
            Float { endianness, unit } => BinarySpecifier::Float {
                endianness: eir_to_target_endianness(endianness),
                unit,
            },
            Bytes { unit } => BinarySpecifier::Bytes { unit },
            Bits { unit } => BinarySpecifier::Bits { unit },
            Utf8 => BinarySpecifier::Utf8,
            Utf16 { endianness } => BinarySpecifier::Utf16 {
                endianness: eir_to_target_endianness(endianness),
            },
            Utf32 { endianness } => BinarySpecifier::Utf32 {
                endianness: eir_to_target_endianness(endianness),
            },
        }
    }
}

#[inline]
fn eir_to_target_endianness(e: libeir_ir::Endianness) -> Endianness {
    match e {
        libeir_ir::Endianness::Big => Endianness::Big,
        libeir_ir::Endianness::Little => Endianness::Little,
        libeir_ir::Endianness::Native => Endianness::Native,
    }
}

include!(concat!(env!("TERM_LIB_OUTPUT_DIR"), "/term_encoding.rs"));

// Type is defined in tablegen in lumen/compiler/Dialect/EIR/IR/EIRBase.td,
// and generated via lumen/compiler/Dialect/Tools/EIREncodingGen.cpp
impl From<libeir_ir::BasicType> for Type {
    fn from(ty: libeir_ir::BasicType) -> Self {
        use libeir_ir::BasicType;
        match ty {
            BasicType::List => Type::List,
            BasicType::ListCell => Type::Cons,
            BasicType::Nil => Type::Nil,
            BasicType::Tuple(arity) => Type::Tuple(arity as u32),
            BasicType::Map => Type::Map,
            BasicType::Number => Type::Number,
            BasicType::Float => Type::Float,
            BasicType::Integer => Type::Integer,
            BasicType::SmallInteger => Type::Fixnum,
            BasicType::BigInteger => Type::BigInt,
        }
    }
}

/// Used to represent a map_update/map_insert operation
#[repr(C)]
pub struct MapUpdate {
    pub loc: LocationRef,
    pub map: ValueRef,
    pub ok: BlockRef,
    pub err: BlockRef,
    pub actionsv: *const MapAction,
    pub actionsc: usize,
}

/// Used to represent a specific update/insert action which
/// occurs as part of a `MapUpdate`
#[derive(Debug)]
#[repr(C)]
pub struct MapAction {
    pub action: MapActionType,
    pub key: ValueRef,
    pub value: ValueRef,
}

/// The type of "put" action, i.e. insertion or update
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub enum MapActionType {
    Insert = 1,
    Update,
}

/// A constant key/value pair used in MLIR attributes
#[repr(C)]
#[derive(Debug)]
pub struct KeyValuePair {
    pub key: AttributeRef,
    pub value: AttributeRef,
}

/// A key/value pair used when constructing maps
#[repr(C)]
#[derive(Debug)]
pub struct MapEntry {
    pub key: ValueRef,
    pub value: ValueRef,
}

/// The MLIR representation of a match pattern
#[repr(C)]
#[derive(Debug)]
pub enum MatchPattern {
    Any,
    Cons,
    Tuple(libc::c_uint),
    MapItem(ValueRef),
    IsType(Type),
    Value(ValueRef),
    Binary {
        size: ValueRef,
        spec: BinarySpecifier,
    },
}

/// The MLIR representation of a specific arm of a match operation,
/// i.e. one pattern with one destination block
#[repr(C)]
#[derive(Debug)]
pub struct MatchBranch {
    pub loc: LocationRef,
    pub dest: BlockRef,
    pub dest_argv: *const ValueRef,
    pub dest_argc: libc::c_uint,
    pub pattern: MatchPattern,
}

/// The MLIR representation of a match operation
#[repr(C)]
#[derive(Debug)]
pub struct MatchOp {
    pub loc: LocationRef,
    pub selector: ValueRef,
    pub branches: *const MatchBranch,
    pub num_branches: libc::c_uint,
}

extern "C" {
    pub fn MLIRCreateModuleBuilder(
        context: ContextRef,
        name: *const libc::c_char,
        loc: SourceLocation,
        target_machine: llvm::target::TargetMachineRef,
    ) -> ModuleBuilderRef;

    #[allow(unused)]
    pub fn MLIRDumpModule(builder: ModuleBuilderRef);

    pub fn MLIRFinalizeModuleBuilder(builder: ModuleBuilderRef) -> ModuleBuilderResult;

    //---------------
    // Locations
    //---------------

    pub fn MLIRCreateLocation(builder: ModuleBuilderRef, loc: SourceLocation) -> LocationRef;

    pub fn MLIRCreateFusedLocation(
        builder: ModuleBuilderRef,
        locs: *const LocationRef,
        num_locs: libc::c_uint,
    ) -> LocationRef;

    #[allow(dead_code)]
    pub fn MLIRUnknownLocation(builder: ModuleBuilderRef) -> LocationRef;

    //---------------
    // Functions
    //---------------

    pub fn MLIRCreateFunction(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        name: *const libc::c_char,
        argv: *const Param,
        argc: libc::c_uint,
        result_type: *const Type,
    ) -> FunctionDeclResult;

    pub fn MLIRAddFunction(builder: ModuleBuilderRef, function: FunctionOpRef);

    pub fn MLIRBuildClosure(builder: ModuleBuilderRef, closure: *const Closure) -> ValueRef;

    pub fn MLIRBuildUnpackEnv(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        env: ValueRef,
        values: *mut ValueRef,
        num_values: libc::c_uint,
    ) -> bool;

    //---------------
    // Blocks
    //---------------

    #[allow(unused)]
    pub fn MLIRGetCurrentBlockArgument(builder: ModuleBuilderRef, id: libc::c_uint) -> ValueRef;

    pub fn MLIRGetBlockArgument(block: BlockRef, id: libc::c_uint) -> ValueRef;

    pub fn MLIRAppendBasicBlock(
        builder: ModuleBuilderRef,
        fun: FunctionOpRef,
        argv: *const Param,
        argc: libc::c_uint,
    ) -> BlockRef;

    pub fn MLIRBlockPositionAtEnd(builder: ModuleBuilderRef, block: BlockRef);

    //---------------
    // Control Flow
    //---------------

    pub fn MLIRBuildBr(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        dest: BlockRef,
        argv: *const ValueRef,
        argc: libc::c_uint,
    );

    pub fn MLIRBuildIf(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: ValueRef,
        yes: BlockRef,
        yes_argv: *const ValueRef,
        yes_argc: libc::c_uint,
        no: BlockRef,
        no_argv: *const ValueRef,
        no_argc: libc::c_uint,
        other: BlockRef,
        other_argv: *const ValueRef,
        other_argc: libc::c_uint,
    );

    pub fn MLIRBuildUnreachable(builder: ModuleBuilderRef, loc: LocationRef);

    pub fn MLIRBuildReturn(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: ValueRef,
    ) -> ValueRef;

    pub fn MLIRBuildThrow(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        kind: ValueRef,
        class: ValueRef,
        reason: ValueRef,
    );

    pub fn MLIRBuildStaticCall(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        name: *const libc::c_char,
        argv: *const ValueRef,
        argc: libc::c_uint,
        is_tail: bool,
        ok_block: BlockRef,
        ok_argv: *const ValueRef,
        ok_argc: libc::c_uint,
        err_block: BlockRef,
        err_argv: *const ValueRef,
        err_argc: libc::c_uint,
    );

    pub fn MLIRBuildGlobalDynamicCall(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        module: ValueRef,
        function: ValueRef,
        argv: *const ValueRef,
        argc: libc::c_uint,
        is_tail: bool,
        ok_block: BlockRef,
        ok_argv: *const ValueRef,
        ok_argc: libc::c_uint,
        err_block: BlockRef,
        err_argv: *const ValueRef,
        err_argc: libc::c_uint,
    );

    pub fn MLIRBuildClosureCall(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        closure: ValueRef,
        argv: *const ValueRef,
        argc: libc::c_uint,
        is_tail: bool,
        ok_block: BlockRef,
        ok_argv: *const ValueRef,
        ok_argc: libc::c_uint,
        err_block: BlockRef,
        err_argv: *const ValueRef,
        err_argc: libc::c_uint,
    );

    //---------------
    // Operations
    //---------------

    pub fn MLIRBuildMatchOp(builder: ModuleBuilderRef, op: MatchOp) -> bool;

    pub fn MLIRBuildTraceCaptureOp(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        dest: BlockRef,
        argv: *const ValueRef,
        argc: libc::c_uint,
    );
    pub fn MLIRBuildTraceConstructOp(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        trace: ValueRef,
    ) -> ValueRef;

    pub fn MLIRBuildMapOp(builder: ModuleBuilderRef, op: MapUpdate);
    pub fn MLIRBuildIsEqualOp(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        lhs: ValueRef,
        rhs: ValueRef,
        is_exact: bool,
    ) -> ValueRef;
    pub fn MLIRBuildIsNotEqualOp(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        lhs: ValueRef,
        rhs: ValueRef,
        is_exact: bool,
    ) -> ValueRef;
    pub fn MLIRBuildLessThanOrEqualOp(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef;
    pub fn MLIRBuildLessThanOp(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef;
    pub fn MLIRBuildGreaterThanOrEqualOp(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef;
    pub fn MLIRBuildGreaterThanOp(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef;
    pub fn MLIRBuildLogicalAndOp(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        argv: *const ValueRef,
        argc: libc::c_uint,
    ) -> ValueRef;
    pub fn MLIRBuildLogicalOrOp(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        argv: *const ValueRef,
        argc: libc::c_uint,
    ) -> ValueRef;

    pub fn MLIRCons(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        head: ValueRef,
        tail: ValueRef,
    ) -> ValueRef;
    pub fn MLIRConstructTuple(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        elements: *const ValueRef,
        num_elements: libc::c_uint,
    ) -> ValueRef;
    pub fn MLIRConstructMap(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        pairs: *const MapEntry,
        num_pairs: libc::c_uint,
    ) -> ValueRef;
    pub fn MLIRBuildBinaryStart(builder: ModuleBuilderRef, loc: LocationRef, cont_block: BlockRef);
    pub fn MLIRBuildBinaryPush(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        bin: ValueRef,
        value: ValueRef,
        size: ValueRef,
        spec: &BinarySpecifier,
        ok_block: BlockRef,
        err_block: BlockRef,
    );
    pub fn MLIRBuildBinaryFinish(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        cont_block: BlockRef,
        bin: ValueRef,
    );
    pub fn MLIRBuildReceiveStart(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        cont_block: BlockRef,
        timeout: ValueRef,
    );
    pub fn MLIRBuildReceiveWait(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        timeout_block: BlockRef,
        check_block: BlockRef,
        receive_ref: ValueRef,
    );
    pub fn MLIRBuildReceiveDone(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        cont_block: BlockRef,
        receive_ref: ValueRef,
        argv: *const ValueRef,
        argc: libc::c_uint,
    );

    //---------------
    // Constants
    //---------------

    pub fn MLIRBuildConstantFloat(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: f64,
    ) -> ValueRef;
    pub fn MLIRBuildFloatAttr(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: f64,
    ) -> AttributeRef;
    pub fn MLIRBuildConstantInt(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: i64,
    ) -> ValueRef;
    pub fn MLIRBuildIntAttr(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: i64,
    ) -> AttributeRef;
    pub fn MLIRBuildConstantBigInt(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: *const libc::c_char,
        width: libc::c_uint,
    ) -> ValueRef;
    pub fn MLIRBuildBigIntAttr(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: *const libc::c_char,
        width: libc::c_uint,
    ) -> AttributeRef;
    pub fn MLIRBuildConstantAtom(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: *const libc::c_char,
        id: u64,
    ) -> ValueRef;
    pub fn MLIRBuildAtomAttr(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: *const libc::c_char,
        id: u64,
    ) -> AttributeRef;
    pub fn MLIRBuildConstantBinary(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: *const libc::c_char,
        size: libc::c_uint,
        header: u64,
        flags: u64,
    ) -> ValueRef;
    pub fn MLIRBuildBinaryAttr(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: *const libc::c_char,
        size: libc::c_uint,
        header: u64,
        flags: u64,
    ) -> AttributeRef;
    pub fn MLIRBuildConstantNil(builder: ModuleBuilderRef, loc: LocationRef) -> ValueRef;
    pub fn MLIRBuildNilAttr(builder: ModuleBuilderRef, loc: LocationRef) -> AttributeRef;
    pub fn MLIRBuildConstantList(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        elements: *const AttributeRef,
        num_elements: libc::c_uint,
    ) -> ValueRef;
    pub fn MLIRBuildListAttr(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        elements: *const AttributeRef,
        num_elements: libc::c_uint,
    ) -> AttributeRef;
    pub fn MLIRBuildConstantTuple(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        elements: *const AttributeRef,
        num_elements: libc::c_uint,
    ) -> ValueRef;
    pub fn MLIRBuildTupleAttr(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        elements: *const AttributeRef,
        num_elements: libc::c_uint,
    ) -> AttributeRef;
    pub fn MLIRBuildConstantMap(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        elements: *const KeyValuePair,
        num_elements: libc::c_uint,
    ) -> ValueRef;
    pub fn MLIRBuildMapAttr(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        elements: *const KeyValuePair,
        num_elements: libc::c_uint,
    ) -> AttributeRef;

    //---------------
    // Type Checking
    //---------------

    pub fn MLIRBuildIsTypeTupleWithArity(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: ValueRef,
        arity: libc::c_uint,
    ) -> ValueRef;
    pub fn MLIRBuildIsTypeList(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: ValueRef,
    ) -> ValueRef;
    pub fn MLIRBuildIsTypeNonEmptyList(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: ValueRef,
    ) -> ValueRef;
    pub fn MLIRBuildIsTypeNil(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: ValueRef,
    ) -> ValueRef;
    pub fn MLIRBuildIsTypeMap(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: ValueRef,
    ) -> ValueRef;
    pub fn MLIRBuildIsTypeNumber(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: ValueRef,
    ) -> ValueRef;
    pub fn MLIRBuildIsTypeFloat(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: ValueRef,
    ) -> ValueRef;
    pub fn MLIRBuildIsTypeInteger(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: ValueRef,
    ) -> ValueRef;
    pub fn MLIRBuildIsTypeFixnum(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: ValueRef,
    ) -> ValueRef;
    pub fn MLIRBuildIsTypeBigInt(
        builder: ModuleBuilderRef,
        loc: LocationRef,
        value: ValueRef,
    ) -> ValueRef;
}
