#ifndef CIR_TYPES_TD
#define CIR_TYPES_TD

include "CIR/Dialect.td"
include "CIR/TypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

// Type constraint accepting any term type
def CIR_AnyTerm : Type<CPred<"::mlir::cir::isTermType($_self)">, "any valid term type">;

// Type constraint accepting any pointer-width type
def CIR_Immediate : Type<CPred<"::mlir::cir::isImmediateType($_self)">, "any type which represents a valid immediate">;

// Type constraint accepting any boxable type
def CIR_Boxable : Type<CPred<"::mlir::cir::isBoxableType($_self)">, "any type which can be boxed">;


// Type constraint for boxed values with an additional constraint on the pointee type
class CIR_BoxOf<Type pointee> : Type<
  And<[CPred<"$_self.isa<::mlir::cir::CIRBoxType>()">,
       SubstLeaves<
         "$_self",
         "$_self.cast<::mlir::cir::CIRBoxType>().getElementType()",
         pointee.predicate>]>,
  "CIR pointer to " # pointee.summary>, SameBuildabilityAs<pointee, "::mlir::cir::CIRBoxType::get(" # pointee.builderCall # ")">;

def CIR_ListType : DialectType<CIR_Dialect,
  And<[CPred<"$_self.isa<::mlir::cir::CIRBoxType>()">,
       SubstLeaves<
         "$_self",
         "$_self.cast<::mlir::cir::CIRBoxType>().getElementType()",
         CPred<"$_self.isa<::mlir::cir::CIRConsType>()">>]>,
  "A boxed cons cell"> {
  let builderCall = "::mlir::cir::CIRBoxType::get($_builder.getType<::mlir::cir::CIRConsType>())";
}

class PtrOf<Type pointee> : Type<
  And<[CPred<"$_self.isa<::mlir::cir::PtrType>()">,
       SubstLeaves<
         "$_self",
         "$_self.cast<::mlir::cir::PtrType>().getElementType()",
         pointee.predicate>]>,
  "raw pointer to " # pointee.summary>, SameBuildabilityAs<pointee, "::mlir::cir::PtrType::get(" # pointee.builderCall # ")">;

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

class CIR_Type<string name, list<Trait> traits = [],
             string baseCppClass = "::mlir::Type">
    : TypeDef<CIR_Dialect, !strconcat("CIR", name), !listconcat(traits, [DeclareTypeInterfaceMethods<TermTypeInterface>]), baseCppClass> {
  let mnemonic = ?;
}

class CIR_BoxedType<string name, list<Trait> traits = [],
                    string baseCppClass = "::mlir::Type">
    : CIR_Type<name, !listconcat(traits, [DeclareTypeInterfaceMethods<BoxedTypeInterface>]), baseCppClass> {
}

class CIR_ImmediateType<string name, list<Trait> traits = [],
                        string baseCppClass = "::mlir::Type">
    : CIR_Type<name, !listconcat(traits, [DeclareTypeInterfaceMethods<ImmediateTypeInterface>, DeclareTypeInterfaceMethods<MemRefElementTypeInterface>]), baseCppClass> {
}

def CIR_NoneType : CIR_ImmediateType<"None"> {
  let mnemonic = "none";
  let summary = "the none type";
  let description = "A type that represents an invalid term, and has a corresponding runtime representation";
}

def CIR_OpaqueTermType : TypeDef<CIR_Dialect, "CIROpaqueTerm", [ImmediateTypeInterface, TermTypeInterface], "::mlir::Type"> {
  let mnemonic = "term";
  let summary = "any term";
  let description = "A type that represents a valid term of unknown concrete type. It can be treated as an immediate";
}

def CIR_NumberType : CIR_Type<"Number"> {
  let mnemonic = "number";
  let summary = "a numeric term";
  let description = "A type that represents a term of any numeric type.";
}

def CIR_IntegerType : CIR_Type<"Integer"> {
  let mnemonic = "integer";
  let summary = "an integer term";
  let description = "A type that represents an integer term of either fixed or arbitrary width.";
}

def CIR_FloatType : CIR_ImmediateType<"Float"> {
  let mnemonic = "float";
  let summary = "a floating-point term";
  let description = "A type that represents a floating-point term. The runtime representation of this varies.";
}

def CIR_AtomType : CIR_ImmediateType<"Atom"> {
  let mnemonic = "atom";
  let summary = "an atom";
  let description = "A type that represents an atom term.";
}

def CIR_BoolType : CIR_ImmediateType<"Bool"> {
  let mnemonic = "bool";
  let summary = "a boolean";
  let description = "A type that represents an atom term of value 'true' or 'false'.";
}

def CIR_IsizeType : CIR_ImmediateType<"Isize"> {
  let mnemonic = "isize";
  let summary = "a small integer";
  let description = [{
    A machine-width integer is defined as being no larger than the size of the pointer type on the target system.
    As a general rule, this corresponds to i32 on 32-bit systems, i64 on 64-bit systems, and so on. There is an
    additional constraint here however, as due to term encoding, we don't actually get the full bitwidth on the
    target. Instead, it's more like 48 bits on a 64-bit system, with anything wider being promoted to BigInt.

    This type is always valid as an immediate, and never heap allocated.
  }];
}

def CIR_BigIntType : CIR_BoxedType<"BigInt"> {
  let mnemonic = "bigint";
  let summary = "a big integer";
  let description = [{
    An arbitrary-width integer here is defined not as a specific size in bits, but rather that the type itself can be of
    any bit width and still be considered the same type as another value of a different bit-width if the values are the
    same. This differs a bit from how MLIR in general defines arbitrary bitness, which considers values with different
    widths to be different types. This makes sense in general, but doesn't properly represent Erlang's semantics around
    integers.

    This type is always heap-allocated, and thus boxed.
  }];
}

def CIR_NilType : CIR_ImmediateType<"Nil"> {
  let mnemonic = "nil";
  let summary = "nil";
  let description = "A type that represents the nil term.";
}

def CIR_ConsType : CIR_BoxedType<"Cons"> {
  let mnemonic = "cons";
  let summary = "a non-empty list term";
  let description = "A type that represents a cons term.";
}

def CIR_MapType : CIR_BoxedType<"Map"> {
  let mnemonic = "map";
  let summary = "a map term";
  let description = "A type that represents a map term.";
}

def CIR_BitsType : CIR_BoxedType<"Bits"> {
  let mnemonic = "bits";
  let summary = "a bitstring term";
  let description = "A type that represents a bitstring term.";
}

def CIR_BinaryType : CIR_BoxedType<"Binary"> {
  let mnemonic = "binary";
  let summary = "a binary term";
  let description = "A type that represents a bitstring evenly divisible into units of 8 bits";
}

def CIR_PidType : CIR_BoxedType<"Pid"> {
  let mnemonic = "pid";
  let summary = "a pid term";
  let description = "A type that represents a local or external process identifier term.";
}

def CIR_PortType : CIR_BoxedType<"Port"> {
  let mnemonic = "port";
  let summary = "a port term";
  let description = "A type that represents a local or external port term.";
}

def CIR_ReferenceType : CIR_BoxedType<"Reference"> {
  let mnemonic = "ref";
  let summary = "a reference term";
  let description = "A type that represents a local or external reference term.";
}

def CIR_BoxType : CIR_ImmediateType<"Box"> {
  let mnemonic = "box";
  let summary = "a boxed term";
  let description = "A type that represents a boxed term";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      assert(isBoxableType(elementType) && "invalid element type, must be boxable");
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

def CIR_FunType : CIR_BoxedType<"Fun", [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
  let mnemonic = "fun";
  let summary = "a closure";
  let description = "A type that represents a closure term";
  let parameters = (ins "FunctionType":$calleeType, "ArrayRef<Type>":$envTypes);
  let builders = [
    TypeBuilder<(ins CArg<"FunctionType">:$calleeType, CArg<"ArrayRef<Type>">:$envTypes), [{
      return $_get($_ctxt, calleeType, envTypes);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genStorageClass = 0;
  let genAccessors = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    unsigned getArity() const;
    unsigned getEnvArity() const { return getEnvTypes().size(); }
    bool isThin() const { return getEnvArity() == 0; }

    void walkImmediateSubElements(function_ref<void(Attribute)> walkAttrsFn,
                                    function_ref<void(Type)> walkTypesFn) const;

  }];
}

def CIR_ProcessType : TypeDef<CIR_Dialect, "CIRProcess"> {
  let mnemonic = "process";
  let summary = "an erlang process";
  let description = "A type that represents the current process state during an invocation of a function";
}

def CIR_ExceptionType : TypeDef<CIR_Dialect, "CIRException"> {
  let mnemonic = "exception";
  let summary = "an erlang exception";
  let description = "A type that represents the exception structure used internally for exception handling";
}

def CIR_TraceType : TypeDef<CIR_Dialect, "CIRTrace"> {
  let mnemonic = "trace";
  let summary = "an erlang exception trace";
  let description = "A type that represents a handle to the raw internal exception trace";
}

def CIR_RecvContextType : TypeDef<CIR_Dialect, "CIRRecvContext"> {
  let mnemonic = "recv.context";
  let summary = "a receive context";
  let description = "A type that represents a handle to a receive state machine context";
}

def CIR_BinaryBuilderType : TypeDef<CIR_Dialect, "CIRBinaryBuilder"> {
  let mnemonic = "bin.builder";
  let summary = "a binary builder result type";
  let description = "A type that represents a handle to state associated with a binary constructor";
}

def CIR_MatchContextType : TypeDef<CIR_Dialect, "CIRMatchContext"> {
  let mnemonic = "match.context";
  let summary = "a match context";
  let description = "A type that represents a handle to state associated with a binary pattern matching operation";
}

def PtrType : TypeDef<CIR_Dialect, "Ptr"> {
  let mnemonic = "ptr";
  let summary = "a raw pointer";
  let description = "A type that represents a raw pointer to another type";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
}

#endif
