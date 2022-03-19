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
  let typeBuilder = "::mlir::IntegerType::get($_type.getContext(), 8)";
  let constBuilderCall = "::mlir::cir::EndiannessAttr::get($_builder.getContext(), $_builder.getI8Type(), $0)";
}


#endif
