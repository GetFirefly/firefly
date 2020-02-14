#ifndef EIR_BASE_TD
#define EIR_BASE_TD

include "mlir/IR/OpBase.td"
// include "mlir/IR/OpAsmInterface.td"

// We need to define the EIR dialect within the ODS framework,
// so that our operations can be defined against it
def eir_Dialect : Dialect {
  let name = "eir";
  let cppNamespace = "eir";
}

//===----------------------------------------------------------------------===//
// Term Representation
//===----------------------------------------------------------------------===//

class eir_EC<string name, int flag> : BitEnumAttrCase<name, flag>;

/// Constants
def eir_EC_ConstNone  : eir_EC<"ConstNone", 0x0000>;
def eir_EC_ConstNil   : eir_EC<"ConstNil", 0x0001>;
def eir_EC_ConstFalse : eir_EC<"ConstFalse", 0x0002>;
def eir_EC_ConstTrue  : eir_EC<"ConstTrue", 0x0004>;

/// Representation
def eir_EC_Arch32     : eir_EC<"Arch32", 0x0008>;
def eir_EC_Arch64     : eir_EC<"Arch64", 0x0010>;
def eir_EC_Arch64n    : eir_EC<"Arch64Nanboxed", 0x0020>;
def eir_EC_Immediate  : eir_EC<"Immediate", 0x0040>;
def eir_EC_Boxed      : eir_EC<"Boxed", 0x0080>;
def eir_EC_Literal    : eir_EC<"Literal", 0x0100>;
def eir_EC_Header     : eir_EC<"Header", 0x0200>;

def eir_EncodingAttr : BitEnumAttr<"Encoding", "term encoding metadata", [
  eir_EC_ConstNone,
  eir_EC_ConstNil,
  eir_EC_ConstFalse,
  eir_EC_ConstTrue,
  eir_EC_Arch32,
  eir_EC_Arch64,
  eir_EC_Arch64n,
  eir_EC_Immediate,
  eir_EC_Boxed,
  eir_EC_Literal,
]> {
  let cppNamespace = "eir::Encoding";
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def FuncTypeAttr      : TypeAttrBase<"FunctionType", "function type">;

class AliasedSymbolRefAttr : Attr<CPred<"$_self.isa<FlatSymbolRefAttr>()">,
                                        "symbol reference attribute"> {
  let storageType = [{ FlatSymbolRefAttr }];
  let returnType = [{ StringRef }];
  let constBuilderCall = "$_builder.getSymbolRefAttr($0)";
}

def eir_GlobalRefAttr : AliasedSymbolRefAttr;
def eir_FuncRefAttr   : AliasedSymbolRefAttr;

def eir_NilAttr : TypeAttrBase<"NilType", "nil type">;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class eir_TermKind<string name, int kind> : I32EnumAttrCase<name, kind>;

def eir_TK_Term    : eir_TermKind<"Term", 0>;
def eir_TK_List    : eir_TermKind<"List", 1>;
def eir_TK_Number  : eir_TermKind<"Number", 2>;
def eir_TK_Integer : eir_TermKind<"Integer", 3>;
def eir_TK_Float   : eir_TermKind<"Float", 4>;
def eir_TK_Atom    : eir_TermKind<"Atom", 5>;
def eir_TK_Boolean : eir_TermKind<"Boolean", 6>;
def eir_TK_Fixnum  : eir_TermKind<"Fixnum", 7>;
def eir_TK_BigInt  : eir_TermKind<"BigInt", 8>;
def eir_TK_Nil     : eir_TermKind<"Nil", 9>;
def eir_TK_Cons    : eir_TermKind<"Cons", 10>;
def eir_TK_Tuple   : eir_TermKind<"Tuple", 11>;
def eir_TK_Map     : eir_TermKind<"Map", 12>;
def eir_TK_Closure : eir_TermKind<"Closure", 13>;
def eir_TK_Binary  : eir_TermKind<"Binary", 14>;
def eir_TK_HeapBin : eir_TermKind<"HeapBin", 15>;
def eir_TK_ProcBin : eir_TermKind<"ProcBin", 16>;
def eir_TK_Box     : eir_TermKind<"Box", 17>;

class eir_TermType<eir_TermKind kind, string description> : DialectType<
    eir_Dialect,
    CPred<"$_self.isa<OpaqueTermType>() && $_self.cast<OpaqueTermType>().isKind(" # kind.value # ")">,
    description> {

  eir_TermKind termKind = kind;
}

def eir_AtomType : eir_TermType<eir_TK_Atom, "atom term">;
def eir_BoolType : eir_TermType<eir_TK_Boolean, "boolean term">;
def eir_FixnumType : eir_TermType<eir_TK_Fixnum, "fixed-width integer term">;
def eir_BigIntType : eir_TermType<eir_TK_BigInt, "big integer term">;
def eir_IntegerType : eir_TermType<eir_TK_Integer, "any integer term">;
def eir_FloatType : eir_TermType<eir_TK_Float, "float term">;
def eir_NilType : eir_TermType<eir_TK_Nil, "nil term">;
def eir_ConsType : eir_TermType<eir_TK_Cons, "cons cell term">;
def eir_ListType : eir_TermType<eir_TK_List, "any list like term">;
def eir_TupleType : eir_TermType<eir_TK_Tuple, "tuple term">;
def eir_MapType : eir_TermType<eir_TK_Map, "map term">;
def eir_ClosureType : eir_TermType<eir_TK_Closure, "closure term">;
def eir_BinaryType : eir_TermType<eir_TK_Binary, "any binary term">;
def eir_HeapBinType : eir_TermType<eir_TK_HeapBin, "heapbin term">;
def eir_ProcBinType : eir_TermType<eir_TK_ProcBin, "procbin term">;
def eir_BoxType : eir_TermType<eir_TK_Box, "boxed term">;

def eir_AnyTerm : Type<CPred<"$_self.isa<eir::TermType>()">>;
def eir_BoolLike : AnyTypeOf<[I1, eir_BoolType]>;
def eir_FixnumLike : AnyTypeOf<[I32, I64, eir_FixnumType]>;
def eir_FloatLike : AnyTypeOf<[F64, eir_FixnumType]>;
def eir_ListLike : AnyTypeOf<[eir_NilType, eir_ConsType, eir_ListType]>;
def eir_BinaryLike : AnyTypeOf<[eir_BinaryType, eir_HeapBinType, eir_ProcBinType]>;

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

def eir_OpInterface : OpInterface<"eir_Op"> {
  let description = [{
    Common interface used for EIR ops
  }];

  let methods = [];
}

def YieldPoint : NativeOpTrait<"YieldPoint">;

//===----------------------------------------------------------------------===//
// Base Operations
//===----------------------------------------------------------------------===//

// Base class for EIR dialect operations.
//
//This operation inherits from the base `Op` class in mlir/IR/OpBase.td, which provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class eir_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<eir_Dialect, mnemonic, !listconcat(traits, [eir_OpInterface])> {

  // For each EIR op, the following static functions need to be defined
  // in Ops.cpp:
  //
  // * static M::ParseResult parse<op-c++-class-name>(M::OpAsmParser &parser,
  //                                                  M::OperationState &result)
  // * static void print(M::OpAsmPrinter &p, <op-c++-class-name> op)
  // * static M::LogicalResult verify(<op-c++-class-name> op)
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
}

#endif
