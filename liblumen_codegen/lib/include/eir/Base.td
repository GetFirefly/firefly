#ifndef EIR_BASE_TD
#define EIR_BASE_TD

// We need to define the EIR dialect within the ODS framework,
// so that our operations can be defined against it
def eir_Dialect : Dialect {
  let name = "eir";
  let cppNamespace = "eir";
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// Base type that represents any type belonging to the EIR or Standard dialects
def eir_Type : Type<CPred<"eir::isa_eir_or_std_type($_self)">,
    "EIR dialect type">;

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

// Check that an op can only be used within the scope of a FuncOp
def InFunctionScope : HasParent<"FuncOp">;

//===----------------------------------------------------------------------===//
// Base Builders
//===----------------------------------------------------------------------===//

class eir_Builder<string builder> {
  string eirBuilder = builder;
}

def eir_ZeroResultOpBuilder : OpBuilder<
  "Builder *, OperationState &result,"
  "ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes = {}",
  [{
    result.addOperands(operands);
    for (auto namedAttr : attributes)
      result.addAttribute(namedAttr.first, namedAttr.second);
  }]>;

def eir_OneResultOpBuilder : OpBuilder<
  "Builder *, OperationState &result, Type resultType,"
  "ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes = {}",
  [{
    if (resultType)
      result.addTypes(resultType);
    result.addOperands(operands);
    for (auto namedAttr : attributes)
      result.addAttribute(namedAttr.first, namedAttr.second);
  }]>;

def eir_RuntimeOpBuilder : OpBuilder<
  "Builder *, OperationState &result, Type resultType, Type errType,"
  "ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes = {}",
  [{
    if (resultType)
      result.addTypes(resultType);
    if (errType)
      result.addTypes(errType);
    result.addOperands(operands);
    for (auto namedAttr : attributes)
      result.addAttribute(namedAttr.first, namedAttr.second);
  }]>;

class eir_TwoBuilders<OpBuilder b1, OpBuilder b2> {
  list<OpBuilder> builders = [b1, b2];
}

def eir_AllocateOpBuilder : OpBuilder<
  "Builder *builder, OperationState &result, Type inType,"
  "ArrayRef<Value *> sizes = {}, ArrayRef<NamedAttribute> attributes = {}",
  [{
    result.addTypes(getRefTy(inType));
    result.addAttribute("in_type", M::TypeAttr::get(inType));
    result.addOperands(sizes);
    for (auto namedAttr : attributes)
      result.addAttribute(namedAttr.first, namedAttr.second);
  }]>;

def eir_NamedAllocateOpBuilder : OpBuilder<
  "Builder *builder, OperationState &result, Type inType, StringRef name,"
  "ArrayRef<Value *> sizes = {}, ArrayRef<NamedAttribute> attributes = {}",
  [{
    result.addTypes(getRefTy(inType));
    result.addAttribute("in_type", M::TypeAttr::get(inType));
    result.addAttribute("name", builder->getStringAttr(name));
    result.addOperands(sizes);
    for (auto namedAttr : attributes)
      result.addAttribute(namedAttr.first, namedAttr.second);
  }]>;


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
    Op<eir_Dialect, mnemonic, traits> {

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

class eir_SimpleOp<string mnemonic, list<OpTrait> traits>
  : eir_Op<mnemonic, traits> {
  let parser = [{
    M::FunctionType type;
    M::OpAsmParser::OperandType operand;
    if (parser.parseOperand(operand) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(opnd, type.getInput(0), result.operands) ||
        parser.addTypesToList(type.getResults(), result.types))
      return M::failure();
    return M::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(*getODSOperands(0).begin());
    p << " : ";
    p.printFunctionalType(getOperation());
  }];
}

// Base class for EIR operations with zero results.
class eir_ZeroResultOp<string mnemonic, list<OpTrait> traits = []> :
    eir_Op<mnemonic, traits>, Results<(outs)> {
  let builders = [eir_ZeroResultOpBuilder];
}

// Base class for EIR operations with one result.
class eir_OneResultOp<string mnemonic, list<OpTrait> traits = []> :
    eir_Op<mnemonic, traits>, Results<(outs eir_Type:$res)> {
  let builders = [eir_OneResultOpBuilder];
}

// Base class for EIR operations with two result values, the "normal"
// result, and an i1 boolean flag that indicates if the operation failed.
class eir_RuntimeOp<string mnemonic, list<OpTrait> traits = []> :
    eir_Op<mnemonic, traits>, Results<(outs eir_Type:$res, I1:$err)> {
  let builders = [eir_RuntimeOpBuilder];
}

class eir_SimpleOneResultOp<string mnemonic, list<OpTrait> traits = []> :
    eir_SimpleOp<mnemonic, traits>, Results<(outs eir_Type:$res)> {
  let builders = [eir_OneResultOpBuilder];
}

class eir_AllocatableBaseOp<string mnemonic, list<OpTrait> traits = []> :
    eir_Op<mnemonic, traits>, Results<(outs eir_Type:$res)> {
  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<BoolAttr>:$target
  );
}

#endif
