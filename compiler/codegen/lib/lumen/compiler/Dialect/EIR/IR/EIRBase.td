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

class BaseAtomAttr : Attr<CPred<"$_self.isa<AtomAttr>()">,
                                "atom attribute"> {
  let storageType = [{ AtomAttr }];
  let returnType = [{ APInt }];
}

class BaseBinaryAttr : Attr<CPred<"$_self.isa<BinaryAttr>()">,
                                  "binary attribute"> {
  let storageType = [{ BinaryAttr }];
  let returnType = [{ StringRef }];
}

class BaseSeqAttr : Attr<CPred<"$_self.isa<SeqAttr>()">,
                               "seq attribute"> {
  let storageType = [{ SeqAttr }];
  let returnType = [{ ArrayRef<Attribute> }];
}

def eir_AtomAttr : BaseAtomAttr;
def eir_BinaryAttr : BaseBinaryAttr;
def eir_SeqAttr : BaseSeqAttr;
def eir_GlobalRefAttr : AliasedSymbolRefAttr;
def eir_FuncRefAttr   : AliasedSymbolRefAttr;

def eir_NilAttr : TypeAttrBase<"NilType", "nil type">;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class eir_TermKind<string name, int kind> : I32EnumAttrCase<name, kind>;

def eir_TK_None    : eir_TermKind<"None", 0>;
def eir_TK_Term    : eir_TermKind<"Term", 1>;
def eir_TK_List    : eir_TermKind<"List", 2>;
def eir_TK_Number  : eir_TermKind<"Number", 3>;
def eir_TK_Integer : eir_TermKind<"Integer", 4>;
def eir_TK_Float   : eir_TermKind<"Float", 5>;
def eir_TK_Atom    : eir_TermKind<"Atom", 6>;
def eir_TK_Boolean : eir_TermKind<"Boolean", 7>;
def eir_TK_Fixnum  : eir_TermKind<"Fixnum", 8>;
def eir_TK_BigInt  : eir_TermKind<"BigInt", 9>;
def eir_TK_Nil     : eir_TermKind<"Nil", 10>;
def eir_TK_Cons    : eir_TermKind<"Cons", 11>;
def eir_TK_Tuple   : eir_TermKind<"Tuple", 12>;
def eir_TK_Map     : eir_TermKind<"Map", 13>;
def eir_TK_Closure : eir_TermKind<"Closure", 14>;
def eir_TK_Binary  : eir_TermKind<"Binary", 15>;
def eir_TK_HeapBin : eir_TermKind<"HeapBin", 16>;
def eir_TK_ProcBin : eir_TermKind<"ProcBin", 17>;
def eir_TK_Box     : eir_TermKind<"Box", 18>;

class eir_TermTypeBase<eir_TermKind kind, string description> : DialectType<
  eir_Dialect,
  CPred<"OpaqueTermType::isTypeKind($_self, TypeKind::" # kind.symbol # ")">,
  description> {
  eir_TermKind termKind = kind;
}

class eir_BuildableTermType<eir_TermKind kind, string description, code builder> :
  eir_TermTypeBase<kind, description>,
  BuildableType<builder> {

}

class eir_TermType<eir_TermKind kind, string description, string typename> :
  eir_BuildableTermType<kind, description, "$_builder.getType<::lumen::eir::" # typename # ">()"> {
}

def eir_NoneType : eir_TermType<eir_TK_None, "none", "NoneType">;
def eir_AtomType : eir_TermType<eir_TK_Atom, "atom", "AtomType">;
def eir_BoolType : eir_TermType<eir_TK_Boolean, "bool", "BooleanType">;
def eir_FixnumType : eir_TermType<eir_TK_Fixnum, "fixnum", "FixnumType">;
def eir_BigIntType : eir_TermType<eir_TK_BigInt, "bixint", "BigIntType">;
def eir_NumberType : eir_TermType<eir_TK_Number, "number", "NumberType">;
def eir_IntegerType : eir_TermType<eir_TK_Integer, "integer", "IntegerType">;
def eir_FloatType : eir_TermType<eir_TK_Float, "float", "FloatType">;
def eir_NilType : eir_TermType<eir_TK_Nil, "nil", "NilType">;
def eir_ConsType : eir_TermType<eir_TK_Cons, "cons", "ConsType">;
def eir_ListType : eir_TermType<eir_TK_List, "list", "ListType">;
def eir_TupleType : eir_TermType<eir_TK_Tuple, "tuple", "TupleType">;
def eir_MapType : eir_TermType<eir_TK_Map, "map", "MapType">;
def eir_ClosureType : eir_TermType<eir_TK_Closure, "closure", "ClosureType">;
def eir_BinaryType : eir_TermType<eir_TK_Binary, "binary", "BinaryType">;
def eir_HeapBinType : eir_TermType<eir_TK_HeapBin, "heapbin", "HeapBinType">;
def eir_ProcBinType : eir_TermType<eir_TK_ProcBin, "procbin", "ProcBinType">;
def eir_BoxType : eir_TermTypeBase<eir_TK_Box, "box">;

def eir_RefType : Type<CPred<"$_self.isa<eir::RefType>()">, "pointer type">;
def eir_PtrType : Type<CPred<"$_self.isa<eir::PtrType>()">, "raw pointer type">;
def eir_ReceiveRefType : Type<CPred<"$_self.isa<eir::ReceiveRefType>()">, "opaque receive reference">;
def eir_AnyTerm : Type<CPred<"$_self.isa<::lumen::eir::TermType>()">, "dynamic term type">,
                   BuildableType<"$_builder.getType<::lumen::eir::TermType>()">;
def eir_AnyType : Type<CPred<"$_self.isa<::lumen::eir::OpaqueTermType>()">, "any term type">,
                   BuildableType<"$_builder.getType<::lumen::eir::TermType>()">;
def eir_BoolLike : AnyTypeOf<[I1, eir_BoolType], "boolean-like type">;
def eir_FixnumLike : AnyTypeOf<[I32, I64, eir_FixnumType], "fixed-width integer type">;
def eir_FloatLike : AnyTypeOf<[F64, eir_FixnumType], "float-like type">;
def eir_ListLike : AnyTypeOf<[eir_NilType, eir_ConsType, eir_ListType], "list-like type">;
def eir_BinaryLike : AnyTypeOf<[eir_BinaryType, eir_HeapBinType, eir_ProcBinType], "binary-like type">;
def eir_PointerLike : AnyTypeOf<[eir_BoxType, eir_RefType, eir_PtrType], "pointer-like type">;

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

def eir_OpInterface : OpInterface<"eir_Op"> {
  let description = [{
    Common interface used for EIR ops
  }];

  let methods = [];
}

def eir_BinaryOperatorInterface : OpInterface<"eir_BinaryOperator"> {
  let description = "Common interface for binary comparison ops";

  let methods = [
    StaticInterfaceMethod<"get the builtin symbol name for this op",
      "StringRef", "builtinSymbol">
  ];
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
