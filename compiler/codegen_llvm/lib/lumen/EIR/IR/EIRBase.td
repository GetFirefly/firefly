#ifndef EIR_BASE_TD
#define EIR_BASE_TD

include "EIRDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Extended Base Attributes/Types/Parameters
//===----------------------------------------------------------------------===//

class I8EnumAttrCase<string sym, int val, string str = sym>
  : IntEnumAttrCaseBase<I8, sym, str, val>;

class I8EnumAttr<string name, string summary,
                 list<I8EnumAttrCase> cases> :
    IntEnumAttr<I8, name, summary, cases> {
  let returnType = cppNamespace # "::" # name;
  let underlyingType = "uint8_t";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getInt())";
  let constBuilderCall = "$_builder.getI8IntegerAttr(static_cast<int8_t>($0))";
}

class OptionalParameter<string type, string desc = ""> :
  AttrOrTypeParameter<"::llvm::Optional<" # type # ">", desc> {
  let allocator = [{
    if (!$_self.hasValue()) {
      $_dst = ::llvm::None;
    } else {
      }] # type # [{ *result = $_allocator.allocate();
      *result = $_self.getValue();

      $_dst = Optional::create(result);
    }
  }];
}

class OptionalSelfAllocationParameter<string type, string desc = ""> :
  AttrOrTypeParameter<"::llvm::Optional<" # type # ">", desc> {
  let allocator = [{
    if (!$_self.hasValue())
      $_dst = ::llvm::None;

    $_dst = $_self.allocateInto($_allocator);
  }];
}

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

// Term Representation
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
def eir_TK_Pid     : eir_TermKind<"Pid", 19>;
def eir_TK_Reference : eir_TermKind<"Reference", 20>;

// The pointer width of the target architecture, if known
class TargetPointerWidthValue<string name, int val> : I8EnumAttrCase<name, val>;

def TargetPointerWidth : I8EnumAttr<"TargetPointerWidth", "target bit-width", [
  TargetPointerWidthValue<"any", 0>,
  TargetPointerWidthValue<"arch32", 32>,
  TargetPointerWidthValue<"arch64", 64>
]> {
  let cppNamespace = "::lumen::eir";
}

// The target instruction set, if known
class TargetArchValue<string name, int val> : StrEnumAttrCase<name, val>;

def TargetArch : StrEnumAttr<"TargetArch", "target architecture", [
  TargetArchValue<"any", 0>,
  TargetArchValue<"aarch64", 1>,
  TargetArchValue<"x86_64", 2>,
  TargetArchValue<"i686", 3>,
  TargetArchValue<"wasm32", 4>
]> {
  let cppNamespace = "::lumen::eir";
}

// A target feature relevant to code generation
class TargetFeatureValue<string name, int val> : StrEnumAttrCase<name, val>;

def TargetFeature : StrEnumAttr<"TargetFeature", "target-specific features", [
  TargetFeatureValue<"nanboxing", 1>
]> {
  let cppNamespace = "::lumen::eir";
}

def TargetFeatureArray : TypedArrayAttrBase<TargetFeature, "array of target features"> {
  let constBuilderCall = "$_builder.getStrArrayAttr($0)";
}

// A complete specification of a target platform
def TargetPlatformAttr : StructAttr<"TargetPlatformAttr", eir_Dialect, [
  StructFieldAttr<"pointerWidth", OptionalAttr<TargetPointerWidth>>,
  StructFieldAttr<"arch", DefaultValuedAttr<TargetArch, "any">>,
  StructFieldAttr<"features", DefaultValuedAttr<TargetFeatureArray, "{}">>
]>;

//===----------------------------------------------------------------------===//
// Type Traits/Interfaces
//===----------------------------------------------------------------------===//

// This interface is shared across all EIR types which are also terms. It provides
// common functionality that is specific to term type interactions and lowerings.
def TermTypeInterface : TypeInterface<"TermTypeInterface"> {
  let description = "This interface is implemented for all types which are valid term types";
  let cppNamespace = "lumen::eir";

  let methods = [
    InterfaceMethod<"Returns the TermKind for this type", "::lumen::eir::TypeKind::Kind", "getTypeKind">,
    InterfaceMethod<"Returns a value indicating if this type is matchable against the given type constraint",
                    "unsigned", "isMatch", (ins "::mlir::Type":$matcher), [{ return ::lumen::eir::isMatch($_type, matcher); }]>,
    InterfaceMethod<"Returns true if this type is can ever compare equal to a value of the given type",
                    "bool", "canTypeEverBeEqual", (ins "::mlir::Type":$other, "bool":$strict), [{ return ::lumen::eir::canTypeEverBeEqual($_type, other, strict); }]>,
    InterfaceMethod<"Returns true if this type is a numeric type", "bool", "isNumber", (ins), [{ return $_type.template hasTrait<::lumen::eir::NumberLike>(); }]>,
    InterfaceMethod<"Returns true if this type is an atom-like type", "bool", "isAtomLike", (ins), [{ return $_type.template hasTrait<::lumen::eir::AtomLike>(); }]>,
    InterfaceMethod<"Returns true if this type is a list-like type", "bool", "isListLike", (ins), [{ return $_type.template hasTrait<::lumen::eir::ListLike>(); }]>,
    InterfaceMethod<"Returns true if this type is an immediate on the given target", "bool", "isImmediate", (ins "::lumen::eir::TargetPlatform":$target)>,
    InterfaceMethod<"Returns true if this type requires boxing on the given target", "bool", "isBoxable", (ins "::lumen::eir::TargetPlatform":$target)>
  ];

  let extraClassDeclaration = [{}];
  let extraTraitClassDeclaration = [{}];
}

class TypeTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::lumen::eir";
}

// This trait definition causes `TermTypeInterface` to be implemented on the type it is attached to
def IsTermType : TypeTrait<"TermTypeInterface::Trait">;

// This trait is implemented on types which are numeric
def NumberLike : TypeTrait<"NumberLike">;

// This trait is implemented on types which are represented as atoms (e.g. booleans)
def AtomLike : TypeTrait<"AtomLike">;

// This trait is implemented on types which are valid list terms
def ListLike : TypeTrait<"ListLike">;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// The base class for all EIR dialect type constraints
class eir_Type<Pred condition, string description, string builder>
  : DialectType<eir_Dialect, condition, description>, BuildableType<builder>;

// The base class for all EIR dialect type definitions
class eir_TypeDef<string name, string short, string desc, list<OpTrait> typeTraits = []> : TypeDef<eir_Dialect, name> {
  let mnemonic = short;
  let description = desc;
  let printer = "$_printer << \"" # short # "\"";
  let parser = [{ return get($_ctxt); }];
  let builderCall = "$_builder.getType<::lumen::eir::" # name # ">()";
  let traits = typeTraits;
}

// A specialization of eir_TypeDef for term types
class eir_TermType<eir_TermKind kind, string mnemonic, string desc, list<OpTrait> typeTraits = []>
        : eir_TypeDef<kind.str, mnemonic, desc, !listconcat([IsTermType], typeTraits)> {
  let extraClassDeclaration = [{
  private:
    const ::lumen::eir::TypeKind::Kind TYPE_KIND = ::lumen::eir::TypeKind::Kind::}] # kind.str # [{;

  public:
    ::lumen::eir::TypeKind::Kind getTypeKind() const { return TYPE_KIND; }
  }];
}

// A type constraint that is true for all types which derive from eir_TermType
def eir_AnyTerm : Type<CPred<"$_self.isa<::lumen::eir::TermTypeInterface>()">, "any term type">;

// Concrete types that have statically known reprentations
def eir_NoneType        : eir_TermType<eir_TK_None, "none", "an unknown/invalid type tag">;
def eir_TermType        : eir_TermType<eir_TK_Term, "term", "an unknown/invalid type tag">;
def eir_AtomType        : eir_TermType<eir_TK_Atom, "atom", "an atom value", [AtomLike]>;
def eir_BooleanType     : eir_TermType<eir_TK_Boolean, "bool", "an atom value equal to true or false", [AtomLike]>;
def eir_IntegerType     : eir_TermType<eir_TK_Integer, "int", "a fixed-width integer value that fits within an immediate", [NumberLike]>;
def eir_FloatType       : eir_TermType<eir_TK_Float, "float", "a nanboxed float value", [NumberLike]>;
def eir_NumberType      : eir_TermType<eir_TK_Number, "number", "any numeric value", [NumberLike]>;
def eir_NilType         : eir_TermType<eir_TK_Nil, "nil", "an empty list", [ListLike]>;
def eir_ConsType        : eir_TermType<eir_TK_Cons, "cons", "a cons cell", [ListLike]>;
def eir_ListType        : eir_TermType<eir_TK_List, "list", "any list value", [ListLike]>;
def eir_MapType         : eir_TermType<eir_TK_Map, "map", "a key/value map">;
def eir_BinaryType      : eir_TermType<eir_TK_Binary, "binary", "any binary value (heap-allocated or ref-counted)">;
def eir_PidType         : eir_TermType<eir_TK_Pid, "pid", "any pid value (local or external)">;
def eir_ReferenceType   : eir_TermType<eir_TK_Reference, "reference", "any reference value (local or external)">;

// Type predicates for types which have multiple concrete variants
def eir_BoolLike     : AnyTypeOf<[eir_BooleanType, I1]>;
def eir_AtomLike     : AnyTypeOf<[eir_AtomType, eir_BooleanType]>;
def eir_ListLike     : AnyTypeOf<[eir_ListType, eir_ConsType, eir_NilType]>;
def eir_NumberLike   : AnyTypeOf<[eir_NumberType, eir_FloatType, eir_IntegerType]>;

// A type constraint that allows a specific type, or an opaque term
class TermOr<Type t> : AnyTypeOf<[t, eir_TermType]>;

// Non-term types present in generated IR
def eir_BinaryBuilderType : eir_TypeDef<"BinaryBuilder", "binary_builder", "a transient reference to a binary under construction">;
def eir_TraceRefType : eir_TypeDef<"TraceRef", "trace_ref", "a captured trace reference">;
def eir_ReceiveRefType : eir_TypeDef<"ReceiveRef", "recv_ref", "a receive session reference">;
def eir_ReceiveStatusType : TypeAlias<UI8, "receive wait status type">;

// Tuple
def eir_TupleType : eir_TermType<eir_TK_Tuple, "tuple", "any tuple value"> {
  let parameters = (ins ArrayRefParameter<"Type">:$types);
  let builders = [
    TypeBuilder<(ins "::mlir::TypeRange":$elementTypes), [{
      return $_get($_ctxt, elementTypes);
    }]>,
    TypeBuilder<(ins), [{
      return $_get($_ctxt, ::mlir::TypeRange());
    }]>
  ];
  let skipDefaultBuilders = 1;

  let printer = [{
    $_printer << "tuple";
    if (size() > 0) {
      $_printer << "<";
      llvm::interleaveComma(getTypes(), $_printer);
      $_printer << ">";
    }
  }];

  let parser = [{
    if ($_parser.parseLess())
      return get(ctxt);

    SmallVector<Type, 2> envTypes;
    while (true) {
      Type ty;
      if ($_parser.parseType(&ty)) {
        // No type, so we must be at the end of the list
        break;
      }
      if ($_parser.parseComma()) {
        // No comma, so we must be at the end of the list
        break;
      }
      envTypes.push_back(ty);
    }

    // We expect '>' at this point
    if ($_parser.parseGreater()) {
      $_parser.emitError($_parser.getCurrentLocation(), "expected '>'");
      return Type();
    }
    return get(ctxt, envTypes);
  }];

  let extraClassDeclaration = [{
    size_t size() const { return getTypes().size(); }

    using iterator = ArrayRef<::mlir::Type>::iterator;
    iterator begin() const { return getTypes().begin(); }
    iterator end() const { return getTypes().end(); }

    Type getType(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getTypes()[index];
    }
  }];
}

def eir_AnyTuple     : Type<CPred<"$_self.isa<::lumen::eir::TupleType>()">, "any tuple type">;

class eir_TupleOf<list<Type> allowedTypes>
  : MixedContainerType<AnyTypeOf<allowedTypes>, eir_AnyTuple.predicate,
                       "$_self.cast<::lumen::eir::TupleType>().getTypes()",
                       "tuple">;

// Closure

def eir_ClosureType : eir_TermType<eir_TK_Closure, "closure", "any closure value"> {
  let parameters = (ins
    OptionalParameter<"FunctionType">:$signatureType,
    ArrayRefParameter<"Type">:$envTypes
  );
  let builders = [
    TypeBuilder<(ins "FunctionType":$sig, "TypeRange":$env), [{
      return $_get($_ctxt, sig, env);
    }]>,
    TypeBuilder<(ins "FunctionType":$sig), [{
      return $_get($_ctxt, sig, TypeRange());
    }]>,
    TypeBuilder<(ins), [{
      return $_get($_ctxt, ::llvm::None, TypeRange());
    }]>
  ];
  let skipDefaultBuilders = 1;

  let printer = [{
    $_printer << "closure";
    auto funTyOpt = getImpl()->getSignatureType();
    auto envTypes = getEnvTypes();
    if (funTyOpt.hasValue()) {
      $_printer << "<";
      $_printer.printType(funTyOpt.getValue());
      if (envTypes.size() > 0) {
        $_printer << ", [";
        llvm::interleaveComma(envTypes, $_printer);
        $_printer << "]";
      }
      $_printer << ">";
    }
  }];

  let parser = [{
    if ($_parser.parseLess())
      return get(ctxt);

    FunctionType funTy;
    if ($_parser.parseType(&funTy)) {
      $_parser.emitError($_parser.getCurrentLocation(), "expected function type");
      return Type();
    }
    if ($_parser.parseComma()) {
      // No env, expect greater
      if ($_parser.parseGreater()) {
        $_parser.emitError($_parser.getCurrentLocation(), "expected '>'");
        return Type();
      }
      return get(ctxt, funTy);
    }
    if ($_parser.parseLSquare()) {
      $_parser.emitError($_parser.getCurrentLocation(), "expected '['");
      return Type();
    }
    SmallVector<Type, 2> envTypes;
    while (true) {
      Type ty;
      if ($_parser.parseType(&ty)) {
        // No type, so we must be at the end of the list
        break;
      }
      if ($_parser.parseComma()) {
        // No comma, so we must be at the end of the list
        break;
      }
      envTypes.push_back(ty);
    }

    // We expect ']>' at this point
    if ($_parser.parseRSquare()) {
      $_parser.emitError($_parser.getCurrentLocation(), "expected ']'");
      return Type();
    }
    if ($_parser.parseGreater()) {
      $_parser.emitError($_parser.getCurrentLocation(), "expected '>'");
      return Type();
    }
    return get(ctxt, funTy, envTypes);
  }];

  let extraClassDeclaration = [{
    // Return size of the closure's environment (in terms)
    size_t getEnvSize() const { return getEnvTypes().size(); }

    // A closure is opaque if we do not know the signature of the closed over function.
    // Closures of this type must be called via the runtime, which verifies that the signature
    // of the callee matches the passed arguments.
    bool isOpaque() const { return getSignatureType().hasValue(); }

    // A closure is thin if it has no environment, i.e. we can call it like a regular function.
    // We do not consider a closure thin if the signature is unknown, as we must still call those
    // via the runtime. This is intended for optimization purposes, as we can eliminate all of the
    // closure setup code if we can transform it into a trivial call.
    bool isThin() const { return getEnvSize() == 0 && !isOpaque(); }

    // Obtain the type of a value in the closure environment, at a given slot
    Type getType(size_t index) const {
      assert(index < getEnvSize() && "invalid index for tuple type");
      return getEnvTypes()[index];
    }
  }];
}

def eir_AnyClosure   : Type<CPred<"$_self.isa<::lumen::eir::ClosureType>()">, "any closure type">;

// Box<T>
def BoxType : eir_TermType<eir_TK_Box, "box", "a boxed term, i.e. a tagged pointer"> {
  let parameters = (ins "Type":$pointeeType);
  let builders = [
    TypeBuilder<(ins "Type":$pointee), [{
      return $_get($_ctxt, pointee);
    }]>,
    TypeBuilder<(ins), [{
      return $_get($_ctxt, TermType::get($_ctxt));
    }]>
  ];
  let skipDefaultBuilders = 1;

  let printer = [{
    $_printer << "box<";
    $_printer.printType(getImpl()->pointee);
    $_printer << ">";
  }];

  let parser = [{
    Type ty;
    if ($_parser.parseLess() || $_parser.parseType(ty) || $_parser.parseGreater()) {
      $_parser.emitError($_parser.getCurrentLocation(), "pointee type expected");
      return Type();
    }
    return get(ctxt, ty);
  }];
}

def eir_AnyBox : Type<CPred<"$_self.isa<::lumen::eir::BoxType>()">, "any boxed reference">;

class BoxOf<Type type> :
    Type<And<[
      CPred<"$_self.isa<::lumen::eir::BoxType>()">,
      SubstLeaves<"$_self", "$_self.cast<::lumen::eir::BoxType>().getPointeeType()",
                  type.predicate>
    ]>, "box<" # type.description # ">"> {
  // Set the builder call if the base type has a builder call.
  string builderCall = !if(!empty(type.builderCall),
                           "", "::lumen::eir::BoxType::get(" # type.builderCall # ")");
}

// Ptr<T>
def PtrType : eir_TypeDef<"Ptr", "ptr", "a raw pointer"> {
  let parameters = (ins "Type":$pointeeType, "size_t":$addressSpace);

  let builders = [
    TypeBuilder<(ins "Type":$pointee, "size_t":$addrspace), [{
      return $_get($_ctxt, pointee, addrspace);
    }]>,
    TypeBuilder<(ins "Type":$pointee), [{
      return $_get($_ctxt, pointee, 0);
    }]>,
    TypeBuilder<(ins), [{
      return $_get($_ctxt, ::mlir::IntegerType::get($_ctxt, 8), 0);
    }]>
  ];
  let skipDefaultBuilders = 1;

  let printer = [{
    $_printer << "ptr<";
    $_printer.printType(getImpl()->pointee);
    auto addrspace = getImpl()->addressSpace;
    if (addrspace != 0) {
      $_printer << ", ";
      $_printer << addrspace;
    }
    $_printer << ">";
  }];

  let parser = [{
    Type pointee;
    size_t addrspace = 0;
    if ($_parser.parseLess() || $_parser.parseType(&pointee)) {
      $_parser.emitError($_parser.getCurrentLocation(), "pointee type expected");
      return Type();
    }
    if ($_parser.parseComma()) {
      if ($_parser.parseGreater()) {
        $_parser.emitError($_parser.getCurrentLocation(), "expected closing '>'");
        return Type();
      }
    } else {
      if ($_parser.parseInteger(&addrspace)) {
        $_parser.emitError($_parser.getCurrentLocation(), "expected address space");
        return Type();
      }
      if ($_parser.parseGreater()) {
        $_parser.emitError($_parser.getCurrentLocation(), "expected closing '>'");
        return Type();
      }
    }
    return get(ctxt, pointee, addrspace);
  }];
}

def eir_AnyPtr : Type<CPred<"$_self.isa<::lumen::eir::PtrType>()">, "any pointer">;

class PtrOf<Type type, int addrspace = 0> :
    Type<And<[
      CPred<"$_self.isa<::lumen::eir::PtrType>()">,
      CPred<"$_self.cast<::lumen::eir::PtrType>().getAddressSpace() == " # addrspace>,
      SubstLeaves<"$_self", "$_self.cast<::lumen::eir::PtrType>().getPointeeType()",
                  type.predicate>
    ]>, "ptr<" # type.description # ", " # addrspace # ">"> {
  // Set the builder call if the base type has a builder call.
  string builderCall = !if(!empty(type.builderCall),
                           "", "::lumen::eir::PtrType::get(" # type.builderCall # ", " # addrspace # ")");
}

// Some type constraints which can be boxed, depending on target
def eir_IntegerLike  : AnyTypeOf<[eir_IntegerType, BoxOf<eir_IntegerType>]>;
def eir_FloatLike    : AnyTypeOf<[eir_FloatType, BoxOf<eir_FloatType>]>;

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

class eir_AttrDef<string name, string short> : AttrDef<eir_Dialect, name> {
  let mnemonic = short;
}

class eir_Attr<Pred condition, string summary> : Attr<condition, summary> {
  let valueType = NoneType;
  let convertFromStorage = "$_self";
}

def eir_StringAttrDef : eir_AttrDef<"String", "str"> {
  let summary = "An attribute containing a string value";
  let parameters = (ins StringRefParameter<"">:$value,
                        AttributeSelfTypeParameter<"">:$type);
  let builders = [
    AttrBuilderWithInferredContext<(ins "StringRef":$bytes,
                                        "Type":$type), [{
      return $_get(type.getContext(), bytes, type);
    }]>,
    AttrBuilder<(ins "StringRef":$bytes), [{
      return $_get($_ctxt, bytes, ::mlir::NoneType::get($_ctxt));
    }]>
  ];

  let extraClassDeclaration = [{
    using ValueType = StringRef;

    bool isPrintable() const;
  }];
  let skipDefaultBuilders = 1;
}

def eir_StringAttr : eir_Attr<CPred<"$_self.isa<::lumen::eir::StringAttr>()">, "string attribute"> {
  let storageType = [{ ::lumen::eir::StringAttr }];
  let returnType = [{ ::lumen::eir::StringAttr }];
}

def eir_SeqAttrDef : eir_AttrDef<"Seq", "seq"> {
  let summary = "An attribute containing a collection of values";
  let parameters = (ins ArrayRefParameter<"::mlir::Attribute">:$value,
                        AttributeSelfTypeParameter<"">:$type);
  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<::mlir::Attribute>":$elements,
                                        "Type":$type), [{
      return $_get(type.getContext(), elements, type);
    }]>,
    AttrBuilder<(ins "ArrayRef<::mlir::Attribute>":$elements), [{
      return $_get($_ctxt, elements, ::mlir::NoneType::get($_ctxt));
    }]>
  ];

  let extraClassDeclaration = [{
    using ValueType = ArrayRef<::mlir::Attribute>;
  }];
  let skipDefaultBuilders = 1;
}

def eir_SeqAttr : eir_Attr<CPred<"$_self.isa<::lumen::eir::SeqAttr>()">, "sequence attribute"> {
  let storageType = [{ ::lumen::eir::SeqAttr }];
  let returnType = [{ ::lumen::eir::SeqAttr }];
}

def eir_APIntAttrDef : eir_AttrDef<"APInt", "int"> {
  let summary = "An attribute containing an APInt value";
  let parameters = (ins "::llvm::APInt":$value);
  let storageClass = "APIntAttributeStorage";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::llvm::APInt":$i,
                                        "Type":$type), [{
      return $_get(type.getContext(), i, type);
    }]>,
    AttrBuilder<(ins "::llvm::APInt":$i), [{
      return $_get($_ctxt, i, ::mlir::NoneType::get($_ctxt));
    }]>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    using ValueType = ::llvm::APInt;

    std::string getValueAsString() const;
    std::string getHash() const;
  }];
}

def eir_APIntAttr : eir_Attr<CPred<"$_self.isa<::lumen::eir::APIntAttr>()">, "arbitrary-precision integer attribute"> {
  let storageType = [{ ::lumen::eir::APIntAttr }];
  let returnType = [{ ::lumen::eir::APIntAttr }];
}

def eir_APFloatAttrDef : eir_AttrDef<"APFloat", "float"> {
  let summary = "An attribute containing an APFloat value";
  let parameters = (ins "::llvm::APFloat":$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "::llvm::APFloat":$f,
                                        "Type":$type), [{
      return $_get(type.getContext(), f, type);
    }]>,
    AttrBuilder<(ins "::llvm::APFloat":$f), [{
      return $_get($_ctxt, f, ::mlir::NoneType::get($_ctxt));
    }]>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    using ValueType = ::llvm::APFloat;
  }];
}

def eir_APFloatAttr : eir_Attr<CPred<"$_self.isa<::lumen::eir::APFloatAttr>()">, "arbitrary-precision floating-point attribute"> {
  let storageType = [{ ::lumen::eir::APFloatAttr }];
  let returnType = [{ ::lumen::eir::APFloatAttr }];
}

class eir_BinaryFlag<string name, int kind> : I8EnumAttrCase<name, kind>;

def eir_BF_Raw           : eir_BinaryFlag<"Raw", 1>;
def eir_BF_Latin1        : eir_BinaryFlag<"Latin1", 2>;
def eir_BF_Utf8          : eir_BinaryFlag<"Utf8", 3>;
def eir_BF_RawLiteral    : eir_BinaryFlag<"RawLiteral", 5>;
def eir_BF_Latin1Literal : eir_BinaryFlag<"Latin1Literal", 6>;
def eir_BF_Utf8Literal   : eir_BinaryFlag<"Utf8Literal", 7>;

def eir_BinaryFlags : I8EnumAttr<"BinaryFlags", "binary header flags", [
  eir_BF_Raw,
  eir_BF_Latin1,
  eir_BF_Utf8,
  eir_BF_RawLiteral,
  eir_BF_Latin1Literal,
  eir_BF_Utf8Literal,
]> {
  let cppNamespace = "::lumen::eir";
}

def eir_BinaryAttrDef : StructAttr<"BinaryAttr", eir_Dialect, [
  // A binary starts with the term header
  StructFieldAttr<"header", eir_APIntAttr>,
  // Followed by the flags indicating the type of binary data
  StructFieldAttr<"flags", DefaultValuedAttr<eir_BinaryFlags, !cast<string>(eir_BF_Raw.value)>>,
  // And finally the binary data itself
  StructFieldAttr<"data", eir_StringAttr>
]>;

def eir_BinaryAttr : eir_Attr<CPred<"$_self.isa<::lumen::eir::BinaryAttr>()">, "binary data attribute"> {
  let storageType = [{ ::lumen::eir::BinaryAttr }];
  let returnType = [{ ::lumen::eir::BinaryAttr }];
}

// Represents the valid status values returned by receive.wait
class eir_ReceiveStatus<string name, int kind> : I8EnumAttrCase<name, kind>;

def eir_RS_Error    : eir_ReceiveStatus<"Error", 0>;
def eir_RS_Ready    : eir_ReceiveStatus<"Ready", 1>;
def eir_RS_Received : eir_ReceiveStatus<"Received", 2>;
def eir_RS_Timeout  : eir_ReceiveStatus<"Timeout", 3>;

def eir_ReceiveStatusAttr : I8EnumAttr<"ReceiveStatus", "receive wait status", [
  eir_RS_Error,
  eir_RS_Ready,
  eir_RS_Received,
  eir_RS_Timeout,
]> {
  let cppNamespace = "::lumen::eir";
}

class eir_ProcessSignal<string name, int kind> : I8EnumAttrCase<name, kind>;

def eir_PS_None           : eir_ProcessSignal<"None", 0>;
def eir_PS_Yield          : eir_ProcessSignal<"Yield", 1>;
def eir_PS_GarbageCollect : eir_ProcessSignal<"GarbageCollect", 2>;
def eir_PS_Error          : eir_ProcessSignal<"Error", 3>;
def eir_PS_Exit           : eir_ProcessSignal<"Exit", 4>;

def eir_ProcessSignalAttr : I8EnumAttr<"ProcessSignal", "process signal type", [
  eir_PS_None,
  eir_PS_Yield,
  eir_PS_GarbageCollect,
  eir_PS_Error,
  eir_PS_Exit,
]> {
  let cppNamespace = "::lumen::eir";
}

// Atoms are represented as a structure of id/name
def eir_AtomAttr : StructAttr<"AtomAttr", eir_Dialect, [
  // The id of the interned string
  StructFieldAttr<"id", eir_APIntAttr>,
  // The interned string
  StructFieldAttr<"value", eir_StringAttr>
]>;

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

def eir_OpInterface : OpInterface<"eir_Op"> {
  let description = [{
    Common interface used for EIR ops
  }];

  let cppNamespace = "::lumen::eir";

  let methods = [];
}

def eir_RuntimeBuiltinOpInterface : OpInterface<"eir_RuntimeBuiltinOp"> {
  let description = "Common interface for operations which correspond to runtime builtins, e.g. __lumen_builtin_*";

  let cppNamespace = "::lumen::eir";

  let methods = [
    StaticInterfaceMethod<"get the builtin symbol name for this op",
      "StringRef", "builtinSymbol">
  ];
}

def eir_IntrinsicOpInterface : OpInterface<"eir_IntrinsicOp"> {
  let description = "Common interface for operations which correspond to compiler intrinsics, e.g. __lumen_*";

  let cppNamespace = "::lumen::eir";

  let methods = [
    StaticInterfaceMethod<"get the intrinsic symbol name for this op",
    "StringRef", "intrinsicSymbol">
  ];
}

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
