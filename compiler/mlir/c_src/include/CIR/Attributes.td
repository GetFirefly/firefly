#ifndef CIR_ATTRIBUTES_TD
#define CIR_ATTRIBUTES_TD

include "CIR/Dialect.td"
include "mlir/IR/AttrTypeBase.td"

class CIR_Attr<string name, list<Trait> traits = [],
               string baseCppClass = "::mlir::Attribute">
    : AttrDef<CIR_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def AtomRefParameter :
    AttrOrTypeParameter<"::mlir::cir::AtomRef", ""> {
  let printer = [{
    $_printer << "id = ";
    $_printer << $_self.symbol;
    $_printer << ", value = \"";
    $_printer << $_self.strref();
    $_printer << "\"";
  }];
  let comparator = [{ $_lhs.symbol == $_rhs.symbol }];
}

def AtomAttr : CIR_Attr<"Atom"> {
  let mnemonic = "atom";
  let summary = "An attribute that holds an atom's unique id and string representation";
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, AtomRefParameter:$value);

  let assemblyFormat = "`<` $value `:` $type `>`";

  let builders = [
    AttrBuilder<(ins "Type":$type, "::mlir::cir::AtomRef":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = ::mlir::cir::AtomRef;
    size_t getSymbol() const { return getValue().symbol; }
    ::llvm::StringRef getName() const { return getValue().strref(); }

  }];
  let skipDefaultBuilders = 1;
  let typeBuilder = "CIRAtomType::get($_type.getContext())";
}

def EndiannessParameter :
    AttrOrTypeParameter<"::mlir::cir::Endianness", ""> {
  let printer = [{
    switch ($_self) {
    case ::mlir::cir::Endianness::Big:
      $_printer << "big";
      break;
    case ::mlir::cir::Endianness::Little:
      $_printer << "little";
      break;
    case ::mlir::cir::Endianness::Native:
      $_printer << "native";
      break;
    }
  }];
  let comparator = [{ $_lhs == $_rhs }];
}

def EndiannessAttr : CIR_Attr<"Endianness"> {
  let mnemonic = "endianness";
  let summary = "An attribute that holds an endianness type";
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, EndiannessParameter:$value);

  let assemblyFormat = "`<` $value `:` $type `>`";

  let extraClassDeclaration = [{
    using ValueType = ::mlir::cir::Endianness;
  }];
  let constBuilderCall = "::mlir::cir::EndiannessAttr::get($_builder.getContext(), $_builder.getI8Type(), $0)";
}

def CIR_NoneAttr : CIR_Attr<"None"> {
  let mnemonic = "none";
  let summary = "An attribute used to indicate a constant none value";

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $type `>`";

  let constBuilderCall = "NoneAttr::get($_builder.getContext(), $_builder.getType<CIRNoneType>())";

  let builders = [
    AttrBuilder<(ins), [{ return $_get($_ctxt, CIRNoneType::get($_ctxt)); }]>,
  ];
}

def CIR_NilAttr : CIR_Attr<"Nil"> {
  let mnemonic = "nil";
  let summary = "An attribute used to indicate a constant nil value";

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $type `>`";

  let constBuilderCall = "NilAttr::get($_builder.getContext(), $_builder.getType<CIRNilType>())";

  let builders = [
    AttrBuilder<(ins), [{ return $_get($_ctxt, CIRNilType::get($_ctxt)); }]>,
  ];
}

def CIR_BoolAttr : CIR_Attr<"CIRBool"> {
  let mnemonic = "bool";
  let summary = "An attribute used to indicate a constant boolean atom value";

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "bool":$value);
  let assemblyFormat = "$value `:` $type";

  let builders = [
    AttrBuilder<(ins "bool":$value), [{ return $_get($_ctxt, CIRBoolType::get($_ctxt), value); }]>,
  ];

  let extraClassDeclaration = [{
    using ValueType = bool;
  }];
}

def CIR_IsizeAttr : CIR_Attr<"Isize"> {
  let mnemonic = "isize";
  let summary = "An attribute used to hold constant isize integer values";

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APInt":$value);
  let assemblyFormat = "$value `:` $type";

  let builders = [
    AttrBuilder<(ins "const APInt &":$value), [{ return $_get($_ctxt, CIRIsizeType::get($_ctxt), value); }]>,
    AttrBuilder<(ins "uint64_t":$value), [{
      APInt apValue(64, value, /*signed=*/false);
      return $_get($_ctxt, CIRIsizeType::get($_ctxt), apValue);
    }]>,
  ];

  let extraClassDeclaration = [{
    using ValueType = APInt;

    int64_t getInt() const;
  }];
}

def CIR_FloatAttr : CIR_Attr<"CIRFloat"> {
  let mnemonic = "float";
  let summary = "An attribute used to hold constant floating-point values";

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, APFloatParameter<"">:$value);
  let assemblyFormat = "$value `:` $type";

  let builders = [
    AttrBuilder<(ins "const APFloat &":$value), [{ return $_get($_ctxt, CIRFloatType::get($_ctxt), value); }]>,
    AttrBuilder<(ins "double":$value), [{
      return $_get($_ctxt, CIRFloatType::get($_ctxt), APFloat(value));
    }]>,
  ];

  let extraClassDeclaration = [{
    using ValueType = APFloat;

    double getValueAsDouble() const;
    static double getValueAsDouble(APFloat val);
  }];
}

#endif
