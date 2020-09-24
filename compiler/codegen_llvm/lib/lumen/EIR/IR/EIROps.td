#ifndef EIR_OPS
#define EIR_OPS

include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "EIRBase.td"

def eir_FuncOp : eir_Op<"func", [
    AutomaticAllocationScope,
    DeclareOpInterfaceMethods<CallableOpInterface>,
    FunctionLike,
    IsolatedFromAbove,
    Symbol,
  ]> {
  let summary = [{function defined in EIR}];
  let description = [{
    Represents a function containing EIR ops and those of compatible dialects.
  }];

  let arguments = (ins
    OptionalAttr<UnitAttr>:$noinline
  );

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name,
      FunctionType type, ArrayRef<NamedAttribute> attrs = {},
      ArrayRef<MutableDictionaryAttr> argAttrs = {}
    }]>,
  ];

  let verifier = ?;

  let extraClassDeclaration = [{
    friend class OpTrait::FunctionLike<FuncOp>;

    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs);

    /// Add an entry block to an empty function and set up the block arguments
    /// to match the signature of the function.
    Block *addEntryBlock();

    Block *addBlock() {
      assert(!empty() && "function should at least have an entry block");
      push_back(new Block());
      return &back();
    }

    /// Returns the type of this function.
    FunctionType getType() {
      return getAttrOfType<TypeAttr>(getTypeAttrName())
          .getValue()
          .cast<FunctionType>();
    }

    /// Hook for OpTrait::FunctionLike, returns the number of function
    /// arguments. Depends on the type attribute being correct as checked by
    /// verifyType.
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }

    /// Hook for OpTrait::FunctionLike, returns the number of function results.
    /// Depends on the type attribute being correct as checked by verifyType.
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present. This can check for preconditions of the
    /// getNumArguments hook not failing.
    LogicalResult verifyType();
  }];
}

def eir_ClosureOp : eir_Op<"closure", []> {
  let summary = "Creates a new closure at the current point in the function";

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs eir_PtrType:$closure);

  let assemblyFormat = [{
    $callee `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, eir::Closure *closure, ValueRange operands = {}
    }], [{
      auto termTy = builder.getType<TermType>();
      auto i8Ty = builder.getIntegerType(8);
      auto i32Ty = builder.getIntegerType(32);
      auto callee = builder.getSymbolRefAttr(StringRef(closure->name));
      result.addAttribute("callee", callee);
      result.addAttribute("module", Attribute::getFromOpaquePointer(closure->module));
      result.addAttribute("arity", builder.getIntegerAttr(i8Ty, closure->arity));
      result.addAttribute("env_len", builder.getIntegerAttr(i8Ty, closure->envLen));
      result.addAttribute("index", builder.getIntegerAttr(i32Ty, closure->index));
      result.addAttribute("old_unique", builder.getIntegerAttr(i32Ty, closure->index));
      StringRef unique(&closure->unique[0], 16);
      result.addAttribute("unique", builder.getStringAttr(unique));
      result.addOperands(operands);
      result.addTypes(PtrType::get(builder.getType<ClosureType>(closure->envLen)));
    }]>
  ];

  let verifier = ?;

  let extraClassDeclaration = [{
    mlir::FlatSymbolRefAttr getCallee() { return getAttrOfType<mlir::FlatSymbolRefAttr>("callee"); }
    mlir::IntegerAttr arityAttr() { return getAttrOfType<mlir::IntegerAttr>("arity"); }
    AtomAttr module() { return getAttrOfType<AtomAttr>("module"); }
    mlir::IntegerAttr envLenAttr() { return getAttrOfType<mlir::IntegerAttr>("env_len"); }
    mlir::IntegerAttr indexAttr() { return getAttrOfType<mlir::IntegerAttr>("index"); }
    mlir::IntegerAttr oldUniqueAttr() { return getAttrOfType<mlir::IntegerAttr>("old_unique"); }
    mlir::StringAttr uniqueAttr() { return getAttrOfType<mlir::StringAttr>("unique"); }

    unsigned arity() { return arityAttr().getValue().getLimitedValue(); }
    unsigned envLen() { return envLenAttr().getValue().getLimitedValue(); }
    unsigned index() { return indexAttr().getValue().getLimitedValue(); }
    unsigned oldUnique() { return oldUniqueAttr().getValue().getLimitedValue(); }
    StringRef unique() { return uniqueAttr().getValue(); }

    bool isAnonymous() { return envLenAttr() != nullptr && envLen() > 0; }
  }];
}

def eir_UnpackEnvOp : eir_Op<"closure.unpack", []> {
  let summary = "Unpacks a value from a closure environment";

  let arguments = (ins eir_PtrType:$env, Confined<I32Attr, [IntNonNegative]>:$index);
  let results = (outs eir_AnyType:$result);

  let assemblyFormat = [{
    $env `[` $index `]` attr-dict `:` functional-type(operands, results)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value env, unsigned index",
    [{
      result.addOperands(env);
      result.addTypes(builder.getType<TermType>());
      result.addAttribute("index", builder.getI32IntegerAttr(index));
    }]>
  ];

  let verifier = ?;

  let extraClassDeclaration = [{
    unsigned envIndex() { return getAttrOfType<mlir::IntegerAttr>("index").getValue().getLimitedValue(); }
  }];
}

//===----------------------------------------------------------------------===//
// Comparisons
//===----------------------------------------------------------------------===//

def eir_IsTypeOp : eir_Op<"typeof", [NoSideEffect]> {
  let summary = "A type checking primitive used in conditional branching.";

  let description = [{
    This operation is used to examine the type of a term value and return a
    boolean or boolean term indicating whether the provided type matches that
    of the value.
  }];

  let arguments = (ins eir_AnyType:$value, TypeAttr:$type);
  let results = (outs I1:$isMatch);

  let assemblyFormat = [{
    $value `is` $type attr-dict `:` functional-type(operands, results)
  }];

  //let hasCanonicalizer = 1;
  let hasFolder = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, "
    "Value value, Type matchType",
    [{
      result.addOperands(value);
      result.addTypes(builder.getI1Type());
      result.addAttribute("type", TypeAttr::get(matchType));
    }]>
  ];

  let extraClassDeclaration = [{
    Type getMatchType() {
      auto attr = getAttrOfType<TypeAttr>("type");
      return attr.getValue();
    }
  }];
}

def eir_IsTupleOp : eir_Op<"is_tuple", [NoSideEffect]> {
  let summary = "A type checking primitive";

  let description = [{
    Returns true if the input value is a tuple. If an (optional) arity
    is also provided, the check also determines if the tuple is of the
    given arity.
  }];

  let arguments = (ins AnyType:$value, Optional<eir_AnyType>:$arity);
  let results = (outs I1:$isMatch);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  //let hasCanonicalizer = 1;
  let hasFolder = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, "
    "Value value, Optional<Value> arity",
    [{
      result.addOperands(value);
      if (arity.hasValue()) {
        result.addOperands(arity.getValue());
      }
      result.addTypes(builder.getI1Type());
    }]>,
    OpBuilder<
    "OpBuilder &builder, OperationState &result, ValueRange operands",
    [{
      result.addOperands(operands);
      result.addTypes(builder.getI1Type());
    }]>
  ];
}

def eir_IsFunctionOp : eir_Op<"is_function", [NoSideEffect]> {
  let summary = "A type checking primitive";

  let description = [{
    Returns true if the input value is a function. If an (optional) arity
    is also provided, the check also determines if the function is of the
    given arity.
  }];

  let arguments = (ins AnyType:$value, Optional<eir_AnyType>:$arity);
  let results = (outs I1:$isMatch);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  //let hasCanonicalizer = 1;
  let hasFolder = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, "
    "Value value, Optional<Value> arity",
    [{
      result.addOperands(value);
      if (arity.hasValue()) {
        result.addOperands(arity.getValue());
      }
      result.addTypes(builder.getI1Type());
    }]>,
    OpBuilder<
    "OpBuilder &builder, OperationState &result, ValueRange operands",
    [{
      result.addOperands(operands);
      result.addTypes(builder.getI1Type());
    }]>
  ];
}

class eir_UnaryOperator<string mnemonic, Type right, Type result, list<OpTrait> traits = []>
    : eir_Op<mnemonic, !listconcat(traits, [NoSideEffect, eir_RuntimeBuiltinOpInterface])> {
  // Defines the name of the builtin associated with this comparison op
  string builtinSymbol = "__lumen_builtin_" # mnemonic;

  let summary = ?;
  let description = ?;

  let arguments = (ins right:$rhs);
  let results = (outs result:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let verifier = ?;

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Type resultTy, Value rhs,"
    "ArrayRef<NamedAttribute> attrs = {}",
    [{
      result.addOperands(rhs);
      result.addTypes(resultTy);
      result.addAttributes(attrs);
    }]>
  ];

  code extendedClassDeclaration = [{}];
  let extraClassDeclaration = !strconcat(!cast<string>(extendedClassDeclaration), "static StringRef builtinSymbol() { return \"", builtinSymbol, "\"; }");
}

class eir_BinaryOperator<string mnemonic, Type left, Type right, Type result, list<OpTrait> traits = []>
    : eir_Op<mnemonic, !listconcat(traits, [NoSideEffect, eir_RuntimeBuiltinOpInterface])> {
  // Defines the name of the builtin associated with this comparison op
  string builtinSymbol = "__lumen_builtin_" # mnemonic;

  let summary = ?;
  let description = ?;

  let arguments = (ins left:$lhs, right:$rhs);
  let results = (outs result:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{
    auto numOperands = getNumOperands();
    if (numOperands != 2)
      return emitOpError("invalid number of operands, expected 2");
    return success();
  }];

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Type resultTy, Value lhs, Value rhs,"
    "ArrayRef<NamedAttribute> attrs = {}",
    [{
      result.addOperands(ValueRange({lhs, rhs}));
      result.addTypes(resultTy);
      result.addAttributes(attrs);
    }]>
  ];

  code extendedClassDeclaration = [{}];
  let extraClassDeclaration = !strconcat(!cast<string>(extendedClassDeclaration), "static StringRef builtinSymbol() { return \"", builtinSymbol, "\"; }");
}

class eir_SpecializedBinaryOperator<string mnemonic, Type left, Type right, Type result, list<OpTrait> traits = []>
    : eir_BinaryOperator<mnemonic, left, right, result, !listconcat(traits, [eir_IntrinsicOpInterface])> {

  // Defines the name of the intrinsic associated with this op
  string intrinsicSymbol = ?;

  let extraClassDeclaration = !strconcat(
    !cast<string>(extendedClassDeclaration),
    "static StringRef builtinSymbol() { return \"", builtinSymbol, "\"; } ",
    "static StringRef intrinsicSymbol() { return \"", intrinsicSymbol, "\"; }"
  );
}

class eir_LogicalOp<Type type, string mnemonic, list<OpTrait> traits = []> :
    eir_BinaryOperator<mnemonic, type, type, eir_BoolType, !listconcat(traits, [NoSideEffect])> {

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value lhs, Value rhs",
    [{ build(builder, result, builder.getType<BooleanType>(), lhs, rhs); }]>,
    OpBuilder<
    "OpBuilder &builder, OperationState &result, ValueRange args",
    [{
      result.addOperands(args);
      result.addTypes(builder.getType<BooleanType>());
    }]>
  ];
}

def eir_LogicalAndOp : eir_LogicalOp<AnyType, "logical.and", [Commutative]> {
  let summary = "logical AND";
  //let hasFolder = 1;
}

def eir_LogicalOrOp : eir_LogicalOp<AnyType, "logical.or", [Commutative]> {
  let summary = "logical OR";
  //let hasFolder = 1;
}

class eir_UnaryComparisonOp<Type type, string mnemonic, list<OpTrait> traits = []> :
    eir_Op<mnemonic, !listconcat(traits, [NoSideEffect])> {
  let description = [{
    Compares a single operand against the specified predicate.
  }];

  let arguments = (ins type:$operand);
  let results = (outs eir_BoolType:$result);

  let verifier = ?;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

class eir_BinaryComparisonOp<Type type, string mnemonic, list<OpTrait> traits = []>
    : eir_BinaryOperator<mnemonic, type, type, eir_BoolType, traits> {

  let summary = "Compares two operands with the specified predicate.";

  let arguments = (ins type:$lhs, type:$rhs, OptionalAttr<UnitAttr>:$is_strict);
  let results = (outs eir_BoolType:$result);

  let assemblyFormat = [{
    `(` operands `)` (`strict` $is_strict^)? attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, ValueRange args",
    [{
      assert(args.size() == 2 && "expected binary operator to receive 2 arguments");
      Value lhs = args[0];
      Value rhs = args[1];
      build(builder, result, lhs, rhs);
    }]>,
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value lhs, Value rhs, bool strict = false",
    [{
      auto boolType = builder.getType<eir::BooleanType>();
      build(builder, result, boolType, lhs, rhs, strict ? builder.getUnitAttr() : nullptr);
    }]>
  ];
}

def eir_CmpEqOp :
    eir_BinaryComparisonOp<AnyType, "cmp.eq", [Commutative]> {
  let summary = "term equality comparison operation";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def eir_CmpLtOp :
    eir_BinaryComparisonOp<eir_AnyType, "cmp.lt"> {
  let summary = "term less-than comparison operation";
  //let hasFolder = 1;
}

def eir_CmpLteOp :
    eir_BinaryComparisonOp<eir_AnyType, "cmp.lte"> {
  let summary = "term less-than-or-equal comparison operation";
  //let hasFolder = 1;
}

def eir_CmpGtOp :
    eir_BinaryComparisonOp<eir_AnyType, "cmp.gt"> {
  let summary = "term greater-than comparison operation";
  //let hasFolder = 1;
}

def eir_CmpGteOp :
    eir_BinaryComparisonOp<eir_AnyType, "cmp.gte"> {
  let summary = "term greater-than-or-equal comparison operation";
  //let hasFolder = 1;
}

class eir_UnaryArithmeticOp<Type type, string mnemonic, list<OpTrait> traits = []>
    : eir_UnaryOperator<mnemonic, type, type, traits> {
  let summary = "Peform the specified arithmetic operation on one operand";

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, ValueRange args",
    [{
      assert(args.size() == 1 && "expected exactly one argument for arithmetic operator");
      Value rhs = args[0];
      Type resultTy = rhs.getType();
      build(builder, result, resultTy, rhs);
    }]>
  ];
}

class eir_BinaryArithmeticOp<Type type, string mnemonic, list<OpTrait> traits = []>
    : eir_BinaryOperator<mnemonic, type, type, type, traits> {
  let summary = "Peform the specified arithmetic operation over two operands";

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, ValueRange args",
    [{
      assert(args.size() == 2 && "expected exactly two arguments for arithmetic operator");
      Value lhs = args[0];
      Value rhs = args[1];
      Type resultTy = lhs.getType();
      build(builder, result, resultTy, lhs, rhs);
    }]>
  ];
}

class eir_SpecializedBinaryArithmeticOp<Type type, string mnemonic, list<OpTrait> traits = []>
    : eir_SpecializedBinaryOperator<mnemonic, type, type, type, traits> {
  let summary = "Peform the specified specializable arithmetic operation over two operands";

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, ValueRange args",
    [{
      assert(args.size() == 2 && "expected exactly two arguments for arithmetic operator");
      Value lhs = args[0];
      Value rhs = args[1];
      Type resultTy = lhs.getType();
      build(builder, result, resultTy, lhs, rhs);
    }]>
  ];
}

def eir_AddOp :
    eir_SpecializedBinaryArithmeticOp<eir_AnyType, "math.add", [Commutative]> {
  let summary = "Addition operator";
  let intrinsicSymbol = "@llvm.sadd.with.overflow";
  let hasFolder = 1;
}

def eir_NegOp :
    eir_UnaryArithmeticOp<eir_AnyType, "math.neg"> {
  let summary = "Negation operator";

  let hasCanonicalizer = 1;
}

def eir_SubOp :
    eir_SpecializedBinaryArithmeticOp<eir_AnyType, "math.sub"> {
  let summary = "Subraction operator";
  let intrinsicSymbol = "@llvm.ssub.with.overflow";
  let hasFolder = 1;
}

def eir_MulOp :
    eir_SpecializedBinaryArithmeticOp<eir_AnyType, "math.mul", [Commutative]> {
  let summary = "Multiplication operator";
  let intrinsicSymbol = "@llvm.smul.with.overflow";
  let hasFolder = 1;
}

def eir_DivOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.div"> {
  let summary = "Integer division operator";
  let hasFolder = 1;
}

def eir_FDivOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.fdiv"> {
  let summary = "Floating-point division operator";
  let hasFolder = 1;
}

def eir_RemOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.rem"> {
  let summary = "Integer remainder operator";
  let hasFolder = 1;
}

def eir_BslOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.bsl"> {
  let summary = "Bitwise shift-left operator";
  let hasFolder = 1;
}

def eir_BsrOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.bsr"> {
  let summary = "Bitwise shift-right operator";
  let hasFolder = 1;
}

def eir_BandOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.band", [Commutative]> {
  let summary = "Bitwise AND operator";
  let hasFolder = 1;
}

def eir_BorOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.bor", [Commutative]> {
  let summary = "Bitwise OR operator";
  let hasFolder = 1;
}

def eir_BxorOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.bxor", [Commutative]> {
  let summary = "Bitwise XOR operator";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Control Flow
//===----------------------------------------------------------------------===//

def eir_BranchOp : eir_Op<"br",
    [DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>, NoSideEffect, Terminator]> {
  let summary = "unconditional branch operation";
  let description = [{
    Represents an unconditional branch operation that branches to a target block
    with the given set of arguments.

    ```
    ^bb0(...):
      eir.br ^bb1(%a)
    ^bb1(%blockArg1: i32):
      ...
   ```
  }];

  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);

  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, Block *dest,
      ValueRange destOperands = {}
    }], [{
      result.addSuccessors(dest);
      result.addOperands(destOperands);
    }]>,
   ];

  // Fully verified by traits
  let verifier = ?;

  let extraClassDeclaration = [{
    Block *getDest() { return getSuccessor(); }
    void setDest(Block *block) { setSuccessor(block); }

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index) { return getOperation()->eraseOperand(index); }
  }];

  let hasCanonicalizer = 1;
  let assemblyFormat = [{
      $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}

def eir_CondBranchOp : eir_Op<"cond_br",
  [AttrSizedOperandSegments, 
   DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
   NoSideEffect, Terminator]> {
  let summary = "conditional branch operation";
  let description = [{
    Represents a conditional branch operation that branches to one of the two
    target blocks with the given set of arguments.

    ```
    ^bb0(...):
      eir.cond_br %condition, ^bb1(%a), ^bb2(%b)
    ^bb1(%blockArg1: i32):
      ...
    ^bb2(%blockArg2: i32):
      ...
   ```
  }];

  let arguments = (ins
    eir_BoolLike:$condition,
    Variadic<AnyType>:$trueDestOperands,
    Variadic<AnyType>:$falseDestOperands
  );

  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, Value condition,
      Block *trueDest, ValueRange trueOperands,
      Block *falseDest, ValueRange falseOperands
    }], [{
      build(builder, result, condition, trueOperands, falseOperands, trueDest, falseDest);
    }]>,
  ];

  // Fully verified by traits
  let verifier = ?;

  let extraClassDeclaration = [{
    /// These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    /// The condition operand is the first operand in the list.
    Value getCondition() { return getOperand(0); }

    /// Return the destination if the condition is true.
    Block *getTrueDest() {
      return getSuccessor(trueIndex);
    }

    /// Return the destination if the condition is false.
    Block *getFalseDest() {
      return getSuccessor(falseIndex);
    }

    /// Accessors for operands to the 'true' destination.
    Value getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, Value value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    operand_range getTrueOperands() { return trueDestOperands(); }

    unsigned getNumTrueOperands()  { return getTrueOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      trueDestOperandsMutable().erase(index);
    }

    /// Accessors for operands to the 'false' destination.
    Value getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, Value value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }

    operand_range getFalseOperands() { return falseDestOperands(); }

    unsigned getNumFalseOperands() { return getFalseOperands().size(); }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      falseDestOperandsMutable().erase(index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 1; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];

  let hasCanonicalizer = 1;
  let assemblyFormat = [{
    $condition `:` type($condition) `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict
  }];
}

class eir_CallBaseOp<string mnemonic, list<OpTrait> traits = []> :
    eir_Op<mnemonic, !listconcat(traits, [CallOpInterface])> {
  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCalleeAttr(); }

    FlatSymbolRefAttr getCalleeAttr() { return getAttrOfType<FlatSymbolRefAttr>("callee"); }
    Attribute getMustTailAttr() { return getAttrOfType<mlir::UnitAttr>("musttail"); }
    Attribute getTailAttr() { return getAttrOfType<mlir::UnitAttr>("tail"); }
  }];
}

def eir_CallOp : eir_CallBaseOp<"call"> {
  let summary = [{call operation}];
  let description = [{
    Calls a function with the given arguments.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$result);

  let assemblyFormat = [{
    $callee `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, eir::FuncOp callee,
      ValueRange operands, ArrayRef<NamedAttribute> attrs = {}
    }], [{
      result.addOperands(operands);
      result.addAttribute("callee", builder.getSymbolRefAttr(callee));
      result.addAttributes(attrs);
      result.addTypes(callee.getType().getResults());
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, FlatSymbolRefAttr callee,
      ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attrs = {}
    }], [{
      result.addOperands(operands);
      result.addAttribute("callee", callee);
      result.addAttributes(attrs);
      result.addTypes(resultTypes);
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef callee,
      ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attrs = {}
    }], [{
      build(builder, result, builder.getSymbolRefAttr(callee),
            resultTypes, operands, attrs);
    }]>,
  ];

  let verifier = ?;
  let hasCanonicalizer = 1;
}

class eir_InvokeBaseOp<string mnemonic, list<OpTrait> traits = []> :
    eir_Op<mnemonic, !listconcat(traits,
      [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>,
       Terminator])> {
  let successors = (successor AnySuccessor:$okDest, AnySuccessor:$errDest);

  string extraExtraClassDeclaration = "";

  let extraClassDeclaration = [{
    /// These are the indices into the dests list.
    enum { okIndex = 0, errIndex = 1 };

    /// Return the destination if the condition is true.
    Block *getOkDest() {
      return getSuccessor(okIndex);
    }

    /// Return the destination if the condition is false.
    Block *getErrDest() {
      return getSuccessor(errIndex);
    }

    /// Accessors for operands to the 'true' destination.
    Value getOkOperand(unsigned idx) {
      assert(idx < getNumOkOperands());
      return getOperand(getOkDestOperandIndex() + idx);
    }

    void setOkOperand(unsigned idx, Value value) {
      assert(idx < getNumOkOperands());
      setOperand(getOkDestOperandIndex() + idx, value);
    }

    operand_range getOkOperands() { return okDestOperands(); }

    unsigned getNumOkOperands()  { return getOkOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseOkOperand(unsigned index)  {
      okDestOperandsMutable().erase(index);
    }

    /// Accessors for operands to the 'false' destination.
    Value getErrOperand(unsigned idx) {
      assert(idx < getNumErrOperands());
      return getOperand(getErrDestOperandIndex() + idx);
    }
    void setErrOperand(unsigned idx, Value value) {
      assert(idx < getNumErrOperands());
      setOperand(getErrDestOperandIndex() + idx, value);
    }

    operand_range getErrOperands() { return errDestOperands(); }

    unsigned getNumErrOperands() { return getErrOperands().size(); }

    /// Erase the operand at 'index' from the false operand list.
    void eraseErrOperand(unsigned index) {
      errDestOperandsMutable().erase(index);
    }

  private:
    /// Get the index of the first call argument operand.
    unsigned getArgOperandIndex() { return 0; }

    /// Get the index of the first true destination operand.
    unsigned getOkDestOperandIndex() {
      return getArgOperandIndex() + getNumArgOperands();
    }

    /// Get the index of the first false destination operand.
    unsigned getErrDestOperandIndex() {
      return getOkDestOperandIndex() + getNumOkOperands();
    }

  public:
  }] # extraExtraClassDeclaration;
}

def eir_InvokeOp : eir_InvokeBaseOp<"invoke", []> {
  let summary = "call operation for targets that may raise exceptions";
  let description = [{
    Calls a function with the given arguments, uses the provided ok/err blocks
    to handle the flow of control upon return of the callee, based on whether
    an exception was raised or not.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    Variadic<AnyType>:$okDestOperands,
    Variadic<AnyType>:$errDestOperands
  );
  let results = (outs);

  let assemblyFormat = [{
    $callee (`(` $operands^ `:` type($operands) `)`)? `to`
      $okDest (`(` $okDestOperands^ `:` type($okDestOperands) `)`)? `unwind`
      $errDest (`(` $errDestOperands^ `:` type($errDestOperands) `)`)?
      attr-dict
  }];

  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result,
      eir::FuncOp callee, ValueRange operands,
      Block *okDest, ValueRange okDestOperands,
      Block *errDest, ValueRange errDestOperands
    }], [{
      build(builder, result, builder.getSymbolRefAttr(callee),
            operands, okDestOperands, errDestOperands, okDest, errDest);
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result,
      FlatSymbolRefAttr callee, ValueRange operands,
      Block *okDest, ValueRange okDestOperands,
      Block *errDest, ValueRange errDestOperands
    }], [{
      build(builder, result, callee,
            operands, okDestOperands, errDestOperands, okDest, errDest);
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result,
      StringRef callee, ValueRange operands,
      Block *okDest, ValueRange okOperands,
      Block *errDest, ValueRange errOperands
    }], [{
      build(builder, result, builder.getSymbolRefAttr(callee),
            operands, okOperands, errOperands, okDest, errDest);
    }]>,
  ];

  let verifier = ?;
  let hasCanonicalizer = 1;

  //let hasCanonicalizer = 1;
  let extraExtraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType();
    FlatSymbolRefAttr getCalleeAttr() { return getAttrOfType<FlatSymbolRefAttr>("callee"); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() { return operands(); }

    unsigned getNumArgOperands() { return getArgOperands().size(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<FlatSymbolRefAttr>("callee");
    }
  }];
}

def eir_LandingPadOp : eir_Op<"landing_pad", []> {
  let summary = "generates a landing pad for an invoke operation";
  let description = [{
    An invoke operation specifies the block where control is transferred when an
    exception occurs. This operation is used to handle the initial transfer of
    control.

    It must be the first operation in its containing block, and is also a terminator.
    Behavior-wise, it acts like a branch operation, which will branch to user code once
    the exception has landed.
  }];

  let arguments = (ins Variadic<AnyType>:$catchClauses);
  let results = (outs eir_AnyType:$kind, eir_AnyType:$reason, eir_TraceRefType:$trace);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let verifier = ?;

  let builders = [
    OpBuilder<
        "OpBuilder &builder, OperationState &result, ValueRange catchClauses", [{
        Type termTy = builder.getType<TermType>();
        Type traceTy = builder.getType<TraceRefType>();
        build(builder, result, ArrayRef<Type>{termTy, termTy, traceTy}, catchClauses);
    }]>,
    OpBuilder<
        "OpBuilder &builder, OperationState &result, Value catchClause", [{
        Type termTy = builder.getType<TermType>();
        Type traceTy = builder.getType<TraceRefType>();
        build(builder, result, ArrayRef<Type>{termTy, termTy, traceTy}, ValueRange{catchClause});
    }]>,
  ];
}

def eir_ReturnOp : eir_Op<"return", [NoSideEffect, Terminator, ReturnLike]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation. For example:

      func @foo() : (i32, f8) {
      ...
      return %0, %1 : i32, f8
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let hasCanonicalizer = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, ValueRange operands = ValueRange()
    }], [{
      result.addOperands(operands);
    }]>
  ];

  let assemblyFormat = [{
    attr-dict ($operands^ `:` type($operands))?
  }];
}

def eir_YieldOp : eir_Op<"yield", [YieldPoint]> {
  let summary = "scheduler yield operation";
  let description = [{
    Yields execution back to the scheduler
  }];

  let verifier = ?;
  let assemblyFormat = [{ attr-dict }];
}

def eir_YieldCheckOp : eir_Op<"yield.check",
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
     NoSideEffect, Terminator]> {
  let summary = "Terminator which branches based on the need for a yield";

  let arguments = (ins
    I32:$maxReductions,
    Variadic<eir_AnyType>:$trueDestOperands,
    Variadic<eir_AnyType>:$falseDestOperands
  );
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result,
      Value maxReductions,
      Block *trueDest, ValueRange trueOperands,
      Block *falseDest, ValueRange falseOperands
    }], [{
      build(builder, result, maxReductions,
            trueOperands, falseOperands, trueDest, falseDest);
    }]>,
  ];

  // Fully verified by traits
  let verifier = ?;

  let extraClassDeclaration = [{
    /// These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    Value getMaxReductions() { return getOperand(0); }

    /// Return the destination if the condition is true.
    Block *getTrueDest() {
      return getSuccessor(trueIndex);
    }

    /// Return the destination if the condition is false.
    Block *getFalseDest() {
      return getSuccessor(falseIndex);
    }

    /// Accessors for operands to the 'true' destination.
    Value getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, Value value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    operand_range getTrueOperands() { return trueDestOperands(); }

    unsigned getNumTrueOperands()  { return getTrueOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      trueDestOperandsMutable().erase(index);
    }

    /// Accessors for operands to the 'false' destination.
    Value getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, Value value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }

    operand_range getFalseOperands() { return falseDestOperands(); }

    unsigned getNumFalseOperands() { return getFalseOperands().size(); }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      falseDestOperandsMutable().erase(index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 0; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];

  // let hasCanonicalizer = 1;
  let assemblyFormat = [{
    $maxReductions `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Error Handling Operations
//===----------------------------------------------------------------------===//

def eir_UnreachableOp : eir_Op<"unreachable", [Terminator]> {
  let summary = "Used to inform the compiler that this point in the CFG is unreachable";

  let description = [{
    Terminates a basic block with the assertion that the end of the block
    will never be reached at runtime.  This instruction can be used
    immediately after a call to the runtime to terminate the program.
  }];

  let verifier = ?;

  let assemblyFormat = [{ attr-dict }];
}

def eir_ThrowOp : eir_Op<"throw", [Terminator]> {
  let summary = "Used to throw an exception at the current point of execution";
  let description = [{
    A corollary to `eir.return`, this function terminates execution of
    the current function, returning control up the stack by unwinding.
  }];
  let results = (outs);
  let arguments = (ins
    eir_AnyType:$kind,
    eir_AnyType:$reason,
    eir_TraceRefType:$trace
  );

  let verifier = ?;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    `(` $kind `,` $reason `,` $trace `)` attr-dict `:` `(` type($kind) `,` type($reason) `,` type($trace) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Miscellaneous Operations
//===----------------------------------------------------------------------===//

def eir_IncrementReductionsOp : eir_Op<"reductions.inc", []> {
  let summary = "Increments the current processes' reduction counter";

  let arguments = (ins DefaultValuedAttr<Confined<I32Attr, [IntNonNegative]>, "1">:$increment);
  let results = (outs);

  let verifier = ?;

  let assemblyFormat = [{ $increment attr-dict }];
}

def eir_CastOp : eir_Op<"cast", [NoSideEffect]> {
  let summary = "Performs a type cast/conversion between the given input and output type";

  let arguments = (ins AnyType:$input, TypeAttr:$from, TypeAttr:$to);
  let results = (outs AnyType:$output);

  let hasFolder = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, Value source, Type destType",
    [{
      result.addOperands(source);
      result.addAttribute("from", TypeAttr::get(source.getType()));
      result.addAttribute("to", TypeAttr::get(destType));
      result.addTypes(destType);
    }]>
  ];

  let assemblyFormat = [{
    $input `from` $from `to` $to attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    Type getSourceType() { return getOperation()->getAttrOfType<TypeAttr>("from").getValue(); }
    Type getTargetType() { return getOperation()->getAttrOfType<TypeAttr>("to").getValue(); }
  }];
}

def eir_MallocOp : eir_Op<"malloc"> {
  let summary = "allocates memory on the process heap";
  let description = [{
    The "malloc" operation allocates a region of memory, as specified by its
    type. For example:

      %0 = eir.malloc() : !eir.box<tuple<4xf32>>

    The malloc op is only valid with boxed aggregate types like tuple/cons, with
    static dimensions

    This operation returns a single ssa value of `box` type, which can be used
    by subsequent load and store operations.

    The optional `alignment` attribute may be specified to ensure that the
    region of memory that will be indexed is aligned at the specified byte
    boundary.

      %0 = eir.malloc() {alignment = 8} : !eir.box<!eir.tuple<4xf32>>

      %0 = eir.malloc(%1) {alignment = 8} : !eir.box<<!eir.tuple<?xf32>>
  }];

  let arguments = (ins Optional<eir_AnyType>:$arity,
                   Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment);
  let results = (outs eir_PtrType:$ptr);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, PtrType ptrType, "
              "mlir::IntegerAttr alignment = mlir::IntegerAttr()",
    [{
       build(builder, result, ptrType, llvm::None, alignment);
    }]>,
    OpBuilder<"OpBuilder &builder, OperationState &result, PtrType ptrType, "
              "Optional<Value> arity, "
              "mlir::IntegerAttr alignment = mlir::IntegerAttr()",
    [{
       result.addTypes(ptrType);
       if (arity.hasValue())
         result.addOperands(arity.getValue());
       if (alignment)
         result.addAttribute(getAlignmentAttrName(), alignment);
    }]>
  ];

  let extraClassDeclaration = [{
    static StringRef getAlignmentAttrName() { return "alignment"; }

    Optional<unsigned> getAlignment() {
      return getAttrOfType<mlir::IntegerAttr>(getAlignmentAttrName())
        .getValue()
        .getLimitedValue();
    }

    Type getAllocType() { return getType(); }
  }];

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    `(` ($arity^ `:` type($arity))? `)` (`align` $alignment^)? attr-dict `:` type($ptr)
  }];
}

def eir_LoadOp : eir_Op<"load", []> {
  let summary = "Load a value from a memory reference";

  let description = [{
    Load a value from a memory reference into a virtual register.  Produces
    an immutable ssa-value of the referent type.
  }];

  let arguments = (ins eir_PointerLike:$ref);
  let results = (outs eir_AnyType:$out);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value refVal",
    [{
      result.addOperands(refVal);
      auto refType = refVal.getType();
      if (auto boxed = refType.dyn_cast_or_null<BoxType>()) {
        result.addTypes(boxed.getBoxedType());
      } else if (auto ptr = refType.dyn_cast_or_null<RefType>()) {
        result.addTypes(ptr.getInnerType());
      } else if (auto ptr = refType.dyn_cast_or_null<PtrType>()){
        result.addTypes(ptr.getInnerType());
      } else {
        result.addTypes(refType);
      }
    }]
  >];

  let verifier = [{
    auto refVal = ref();
    if (!refVal)
      return emitError("referenced value must not be null");

    auto refType = refVal.getType();
    if (refType.isa<BoxType>() || refType.isa<RefType>() || refType.isa<PtrType>())
      return success();

    return emitOpError("invalid input type, must be pointer-like");
  }];

  let assemblyFormat = [{
    `(` $ref `:` type($ref) `)` attr-dict `:` type($out)
  }];
}

def eir_GetElementPtrOp : eir_Op<"getelementptr", [NoSideEffect]> {
  let summary = "An aggregate access primitive used to calculate memory addresses to an element of an aggregate type";
  let description = [{
    An aggregate access primitive used to calculate memory addresses to an element of an aggregate type
  }];

  let arguments = (ins eir_PtrType:$base, IndexAttr:$index);
  let results = (outs eir_PointerLike:$ref);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value base, int64_t index",
    [{
      auto indexTy = builder.getIndexType();
      result.addAttribute("index", builder.getIntegerAttr(indexTy, index));
      result.addOperands(base);

      auto valueType = base.getType();
      if (auto ptrType = valueType.dyn_cast_or_null<PtrType>()) {
        auto innerType = ptrType.getInnerType();
        if (auto tupleTy = innerType.dyn_cast_or_null<TupleType>()) {
          if (tupleTy.hasStaticShape() && tupleTy.getArity() > index) {
            result.addTypes(builder.getType<PtrType>(tupleTy.getElementType(index)));
          } else {
            result.addTypes(builder.getType<PtrType>(builder.getType<TermType>()));
          }
        } else if (innerType.isa<ConsType>()) {
          result.addTypes(builder.getType<PtrType>(builder.getType<TermType>()));
        } else {
          result.addTypes(builder.getType<NoneType>());
        }
      } else {
        result.addTypes(builder.getType<NoneType>());
      }
    }]>
  ];

  let verifier = [{
    if (getType().isa<NoneType>())
      return emitOpError("unable to determine pointee type");

    if (!base().getType().isa<PtrType>())
      return emitOpError("invalid source operand, expected pointer");
  }];

  let assemblyFormat = [{
    $base `[` $index `]` attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    static StringRef getIndexAttrName() { return "index"; }

    uint64_t getIndex() { return getAttrOfType<mlir::IntegerAttr>("index").getValue().getLimitedValue(); }
  }];
}

def eir_PrintOp : eir_Op<"print"> {
  let summary = "intrinsic printing operation";
  let description = [{
    Prints to standard output using the provided format string and operands.
  }];

  let arguments = (ins eir_AnyType:$thing);
  let results = (outs eir_AnyTerm:$result);

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value arg",
    [{
      build(builder, result, builder.getType<TermType>(), arg);
    }]>,
    OpBuilder<
    "OpBuilder &builder, OperationState &result, ValueRange args",
    [{
      build(builder, result, builder.getType<TermType>(), args.front());
    }]>
  ];

  let verifier = ?;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

def eir_NullOp : eir_Op<"null", [NoSideEffect]> {
  let summary = "Constructs a constant null value";

  let arguments = (ins);
  let results = (outs AnyType);

  let assemblyFormat = [{
    attr-dict `:` type(results)
  }];

  let verifier = ?;
}

class eir_ConstantOp<Type type, Attr attr, string mnemonic, list<OpTrait> traits = []>
    : eir_Op<mnemonic, !listconcat(traits, [NoSideEffect])> {

  let summary = "Constructs a constant term value";

  let arguments = (ins attr:$value);
  let results = (outs type:$result);

  // Require each constant op to define its builders
  let skipDefaultBuilders = 1;

  list<OpBuilder> extraBuilders = [];

  let builders = !listconcat([
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>
  ], extraBuilders);

  // Verification is specified via traits, or per-operation
  let verifier = ?;

  let assemblyFormat = [{
    $value type(results) attr-dict
  }];

  // This can be used to provide extra functionality in the class declaration
  code extendedClassDeclaration = [{}];

  let extraClassDeclaration = !strconcat(!cast<string>(extendedClassDeclaration), !cast<string>([{
    Attribute getValue() { return getAttr("value"); }
    void setValue(Attribute val) { setAttr("value", val); }
  }]));

  //let hasFolder = 1;
}

def eir_ConstantFloatOp : eir_ConstantOp<eir_FloatLike, APFloatAttr, "constant.float", [ConstantLike]> {
  let extraBuilders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, APFloat &value",
    [{
      auto type = builder.getType<eir::FloatType>();
      auto attr = APFloatAttr::get(builder.getContext(), value);
      build(builder, result, type, attr);
    }]>
  ];

  let hasFolder = 1;
}

def eir_ConstantIntOp : eir_ConstantOp<eir_FixnumLike, FixnumAttr, "constant.int", [ConstantLike]> {
  let extraBuilders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, int64_t value",
    [{
      auto type = builder.getType<FixnumType>();
      auto n = static_cast<uint64_t>(value);
      APInt ap(64, n, /*isSigned=*/true);
      auto attr = APIntAttr::get(builder.getContext(), type, ap);
      build(builder, result, type, attr);
    }]>,
    OpBuilder<
    "OpBuilder &builder, OperationState &result, APInt &value",
    [{
      auto type = builder.getType<FixnumType>();
      auto attr = APIntAttr::get(builder.getContext(), type, value);
      build(builder, result, type, attr);
    }]>
  ];

  let hasFolder = 1;
}

def eir_ConstantBigIntOp : eir_ConstantOp<eir_BoxType, BigIntAttr, "constant.bigint", [ConstantLike]> {
  let extraBuilders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, APInt &value",
    [{
      auto bigIntType = builder.getType<BigIntType>();
      auto type = builder.getType<eir::BoxType>(bigIntType);
      auto attr = APIntAttr::get(builder.getContext(), bigIntType, value);
      build(builder, result, type, attr);
    }]>
  ];

  let hasFolder = 1;
}

def eir_ConstantBoolOp : eir_ConstantOp<eir_BoolLike, BoolAttr, "constant.bool", [ConstantLike]> {
  let extraBuilders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, bool truth",
    [{
      build(builder, result, builder.getType<BooleanType>(), builder.getBoolAttr(truth));
    }]>,
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Type type, bool truth",
    [{
      build(builder, result, type, builder.getBoolAttr(truth));
    }]>,
  ];

  let hasFolder = 1;
}

def eir_ConstantAtomOp : eir_ConstantOp<eir_AtomType, AtomAttr, "constant.atom", [ConstantLike]> {
  let extraBuilders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, APInt id, StringRef name",
    [{
      auto type = builder.getType<AtomType>();
      auto attr = AtomAttr::get(builder.getContext(), id, name);
      build(builder, result, type, attr);
    }]>,
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Type type, APInt id, StringRef name",
    [{
      auto attr = AtomAttr::get(builder.getContext(), id, name);
      build(builder, result, type, attr);
    }]>
  ];

  let hasFolder = 1;
}

def eir_ConstantBinaryOp : eir_ConstantOp<eir_BoxType, BinaryAttr, "constant.binary"> {
  let extraBuilders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result,"
    "StringRef value, uint64_t header, uint64_t flags",
    [{
      auto binType = builder.getType<eir::BinaryType>();
      auto type = builder.getType<eir::BoxType>(binType);
      auto attr = BinaryAttr::get(type, value, header, flags);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantNilOp : eir_ConstantOp<eir_AnyType, TypeAttr, "constant.nil", [ConstantLike]> {
  let extraBuilders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Type type",
    [{
      auto attr = TypeAttr::get(type);
      build(builder, result, type, attr);
    }]>,
    OpBuilder<
    "OpBuilder &builder, OperationState &result",
    [{
      auto type = builder.getType<NilType>();
      auto attr = TypeAttr::get(type);
      build(builder, result, type, attr);
    }]>
  ];

  let assemblyFormat = [{
    type(results) attr-dict
  }];

  let hasFolder = 1;
}

def eir_ConstantNoneOp : eir_ConstantOp<eir_AnyTerm, TypeAttr, "constant.none", [ConstantLike]> {
  let extraBuilders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Type type",
    [{
      auto attr = TypeAttr::get(type);
      build(builder, result, type, attr);
    }]>,
    OpBuilder<
    "OpBuilder &builder, OperationState &result",
    [{
      auto type = builder.getType<eir::NoneType>();
      build(builder, result, type);
    }]>
  ];

  let assemblyFormat = [{
    type(results) attr-dict
  }];

  let hasFolder = 1;
}

def eir_ConstantTupleOp : eir_ConstantOp<eir_BoxType, SeqAttr, "constant.tuple"> {
  let extraBuilders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, ArrayRef<Attribute> elements",
    [{
      std::vector<Type> types;
      types.reserve(elements.size());
      for (auto element : elements) {
        if (auto typeAttr = element.dyn_cast_or_null<TypeAttr>()) {
            types.push_back(typeAttr.getValue());
            continue;
        }
        types.push_back(element.getType());
      }
      auto tupleType = builder.getType<eir::TupleType>(types);
      auto type = builder.getType<eir::BoxType>(tupleType);
      auto attr = SeqAttr::get(type, elements);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantListOp : eir_ConstantOp<eir_BoxType, SeqAttr, "constant.list"> {
  let extraBuilders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, ArrayRef<Attribute> elements",
    [{
      auto consType = builder.getType<ConsType>();
      auto type = builder.getType<eir::BoxType>(consType);
      auto attr = SeqAttr::get(type, elements);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantMapOp : eir_ConstantOp<eir_BoxType, SeqAttr, "constant.map"> {
  let extraBuilders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, ArrayRef<Attribute> elements",
    [{
      auto mapType = builder.getType<MapType>();
      auto type = builder.getType<eir::BoxType>(mapType);
      auto attr = SeqAttr::get(type, elements);
      build(builder, result, type, attr);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Constructor Operations
//===----------------------------------------------------------------------===//

def eir_ConsOp : eir_Op<"cons"> {
  let summary = "Cons operation for constructing list cells.";
  let description = [{
    List construction primitive. Constructs a new list cell from head and tail terms.
  }];

  let arguments = (ins eir_AnyType:$head, eir_AnyType:$tail, UnitAttr:$alloca);
  let results = (outs eir_BoxType:$out);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, Value head, Value tail, bool alloca = false",
    [{
      result.addOperands(head);
      result.addOperands(tail);
      result.addTypes(BoxType::get(builder.getType<ConsType>()));
      if (alloca)
        result.addAttribute("alloca", builder.getUnitAttr());
    }]>
  ];

  let verifier = ?;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    `(` operands `)` (`alloca` $alloca^)? attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    bool useAlloca() { return getAttrOfType<mlir::UnitAttr>("alloca") != nullptr; }
  }];
}

def eir_ListOp : eir_Op<"list"> {
  let summary = "List constructor";
  let description = [{
    List construction primitive. Constructs a new list term from a list of elements.
  }];

  let arguments = (ins Variadic<eir_AnyType>:$elements, UnitAttr:$alloca);
  let results = (outs eir_BoxType:$out);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, ArrayRef<Value> elements, bool alloca = false",
    [{
      result.addOperands(elements);
      auto listType = builder.getType<eir::ListType>();
      result.addTypes(BoxType::get(listType));
      if (alloca)
        result.addAttribute("alloca", builder.getUnitAttr());
    }]>
  ];

  let verifier = ?;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    `[` operands `]` (`alloca` $alloca^)? attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    bool useAlloca() { return getAttrOfType<mlir::UnitAttr>("alloca") != nullptr; }
  }];
}

def eir_TupleOp : eir_Op<"tuple"> {
  let summary = "Tuple constructor";
  let description = [{
    Tuple construction primitive. Constructs a new tuple term from a list of elements.
  }];

  let arguments = (ins Variadic<AnyType>:$elements, UnitAttr:$alloca);
  let results = (outs eir_BoxType:$out);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, ArrayRef<Value> elements, bool alloca = false",
    [{
      SmallVector<Type, 1> elementTypes;
      for (auto val : elements) {
        auto valType = val.getType();
        if (valType.isa<OpaqueTermType>())
          elementTypes.push_back(valType);
        else if (valType.isInteger(1))
          elementTypes.push_back(builder.getType<AtomType>());
        else
          elementTypes.push_back(builder.getType<TermType>());
      }
      result.addOperands(elements);
      auto tupleType = builder.getType<eir::TupleType>(elementTypes);
      result.addTypes(BoxType::get(tupleType));
      if (alloca)
        result.addAttribute("alloca", builder.getUnitAttr());
    }]>
  ];

  let verifier = ?;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    `{` operands `}` (`alloca` $alloca^)? attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    bool useAlloca() { return getAttrOfType<mlir::UnitAttr>("alloca") != nullptr; }
  }];
}

def eir_TraceCaptureOp : eir_Op<"trace_capture"> {
  let summary = "Captures the current stack trace";
  let description = [{
    This operation is called when raising an error to inform the process
    to capture and store the current stack trace until a landing pad is
    reached.

    Meant to be used in conjunction with `trace_construct` to access the
    captured trace as a term.
  }];

  let arguments = (ins);
  let results = (outs eir_TraceRefType:$capture);

  let verifier = ?;

  let assemblyFormat = [{
    attr-dict `:` type($capture)
  }];
}

def eir_TracePrintOp : eir_Op<"trace_print"> {
  let summary = "Prints a captured stack trace";
  let description = [{a debugging tool for internal use}];

  let arguments = (ins eir_AnyTerm:$kind, eir_AnyTerm:$reason, eir_TraceRefType:$traceRef);
  let results = (outs);

  let verifier = ?;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def eir_TraceConstructOp : eir_Op<"trace_construct"> {
  let summary = "Constructs the current stack trace as a new SSA-value";
  let description = [{
    This operation should be called prior to generating an error which
    needs the current stack trace. The value returned will be an opaque
    term from the perspective of MLIR, but the runtime can use it like
    a pointer to the raw trace.

        %0 = eir.trace_construct
  }];

  let arguments = (ins eir_TraceRefType:$traceRef);
  let results = (outs eir_BoxType:$trace);

  let verifier = ?;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, Value traceRef",
    [{
      result.addOperands(traceRef);
      result.addTypes(BoxType::get(builder.getType<ConsType>()));
    }]>
  ];
}

def eir_MapOp : eir_Op<"map.new"> {
  let summary = "Map constructor";
  let description = [{
    Map construction primitive. Creates a new map from a list of key/value pairs.
  }];

  let arguments = (ins Variadic<eir_AnyType>:$args);
  let results = (outs eir_BoxType:$out, eir_AnyType:$err);

  let verifier = ?;
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, ArrayRef<Value> elements",
    [{
      result.addOperands(elements);
      result.addTypes(BoxType::get(builder.getType<MapType>()));
    }]>,
    OpBuilder<"OpBuilder &builder, OperationState &result, ArrayRef<eir::MapEntry> entries",
    [{
      for (auto &entry : entries) {
        result.addOperands(Value::getFromOpaquePointer(entry.key));
        result.addOperands(Value::getFromOpaquePointer(entry.value));
      }
      result.addTypes(BoxType::get(builder.getType<MapType>()));
    }]>
  ];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def eir_MapInsertOp : eir_Op<"map.insert", []> {
  let summary = "Inserts a new element in a map";
  let description = [{
    Performs an insert of one or more key/value pairs into a map.

    It is considered an error to try and insert a key that already exists.

    The result of the operation is the updated term as a new SSA value, and
    a second value which is the error flag. This flag is set if the operation
    fails due to a runtime error. If an error occurs, the updated term SSA value
    is undefined.

        %0, %err = eir.map_insert %map, %k, %v : (!eir.box<!eir.map>, !eir.fixnum, !eir.fixnum) -> (!eir.box<!eir.map>, i1)
  }];

  let arguments = (ins eir_AnyTerm:$map, eir_AnyTerm:$key, eir_AnyTerm:$val);
  let results = (outs eir_BoxType:$newMap, I1:$successFlag);

  let verifier = ?;
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value map, Value key, Value value",
    [{
      auto mapType = builder.getType<BoxType>(builder.getType<MapType>());
      auto termType = builder.getType<TermType>();
      auto i1Ty = builder.getI1Type();
      result.addTypes({mapType, i1Ty});
      result.addOperands({map, key, value});
    }]>
  ];

  let assemblyFormat = [{
    $map `(` $key `,` $val `)` attr-dict `:` functional-type(operands, results)
  }];
}

def eir_MapUpdateOp : eir_Op<"map.update", []> {
  let summary = "Update an element in a map";
  let description = [{
    Performs an update of one or more keys in a map.

    The result of the operation is the updated term as a new SSA value, and
    a second value which is the error flag. This flag is set if the operation
    fails due to a runtime error. If an error occurs, the updated term SSA value
    is undefined.

    ## Example

        %0, %err = eir.map_update %map, %k, %v : (!eir.box<!eir.map>, !eir.fixnum, !eir.fixnum) -> (!eir.box<!eir.map>, i1)
  }];

  let arguments = (ins eir_AnyTerm:$map, eir_AnyTerm:$key, eir_AnyTerm:$val);
  let results = (outs eir_BoxType:$newMap, I1:$successFlag);

  let verifier = ?;
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value map, Value key, Value value",
    [{
      auto mapType = builder.getType<BoxType>(builder.getType<MapType>());
      auto termType = builder.getType<TermType>();
      auto i1Ty = builder.getI1Type();
      result.addTypes({mapType, i1Ty});
      result.addOperands({map, key, value});
    }]>
  ];

  let assemblyFormat = [{
    $map `(` $key `,` $val `)` attr-dict `:` functional-type(operands, results)
  }];
}

def eir_MapContainsKeyOp : eir_Op<"map.contains", [NoSideEffect]> {
  let summary = "Returns a boolean indicating whether the given term is a key in the given map";

  let arguments = (ins eir_AnyTerm:$map, eir_AnyTerm:$key);
  let results = (outs I1:$result);

  let verifier = ?;
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value map, Value key",
    [{
      auto i1Ty = builder.getI1Type();
      result.addTypes({i1Ty});
      result.addOperands({map, key});
    }]>
  ];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def eir_MapGetKeyOp : eir_Op<"map.get", []> {
  let summary = "Returns the term associated with the given key in the given map";

  let arguments = (ins eir_AnyTerm:$map, eir_AnyTerm:$key);
  let results = (outs eir_AnyTerm:$result);

  let verifier = ?;
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value map, Value key",
    [{
      auto termType = builder.getType<TermType>();
      result.addTypes(termType);
      result.addOperands({map, key});
    }]>
  ];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def eir_BinaryStartOp : eir_Op<"binary.start"> {
  let summary = "Starts construction of a new binary";
  let arguments = (ins);
  let results = (outs eir_AnyType:$binRef);

  let verifier = ?;

  let assemblyFormat = [{
    attr-dict `:` type(results)
  }];
}


def eir_BinaryFinishOp : eir_Op<"binary.finish"> {
  let summary = "Finishes construction of a new binary";
  let arguments = (ins eir_AnyType:$binRef);
  let results = (outs eir_BoxType:$bin);

  let verifier = ?;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, Value binRef",
    [{
      auto binaryTy = builder.getType<BoxType>(builder.getType<BinaryType>());
      result.addTypes(binaryTy);
      result.addOperands(binRef);
    }]>
  ];
}

def eir_BinaryPushOp : eir_Op<"binary.push"> {
  let summary = "Pushes a value into a binary based on the given specifier";
  let description = [{
    Used to construct a binary piece by piece.

    Each invocation appends a value to the binary based on a binary specification,
    provided as attributes to the operation.

        %0, %err = eir.binary_push(%bin, %val) { type = integer, signed = true, endian = big, unit = 1 }
        %0, %err = eir.binary_push(%bin, %val) { type = bytes, unit = 1 }, %size : !eir.fixnum
  }];

  let arguments = (ins eir_AnyType:$bin, eir_AnyType:$value, Optional<eir_AnyType>:$size);
  let results = (outs eir_AnyType:$newBin, I1:$successFlag);

  let verifier = ?;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value bin, Value value, Value size, ArrayRef<NamedAttribute> attrs",
    [{
      if (size == nullptr) {
        result.addOperands({bin, value});
      } else {
        result.addOperands({bin, value, size});
      }
      auto binTy = bin.getType();
      auto i1Ty = builder.getI1Type();
      result.addTypes({binTy, i1Ty});
      result.addAttributes(attrs);
    }]>
  ];
}

class eir_BinaryMatchOp<string mnemonic, list<OpTrait> traits = []>
    : eir_Op<mnemonic, !listconcat(traits, [eir_RuntimeBuiltinOpInterface])> {
  // Defines the name of the builtin associated with this comparison op
  string builtinSymbol = "__lumen_builtin_" # mnemonic;

  let summary = ?;
  let description = ?;

  let arguments = ?;
  let results = (outs eir_AnyTerm:$matched, eir_AnyTerm:$rest, I1:$successFlag);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let verifier = ?;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    mlir::BoolAttr isSignedAttr() { return getAttrOfType<mlir::BoolAttr>("is_signed"); }
  }] # "static StringRef builtinSymbol() { return \"" # builtinSymbol # "\"; }";
}

def eir_BinaryMatchRawOp : eir_BinaryMatchOp<"binary.match.raw", []> {
  let summary = "Binary pattern match against raw bits/bytes";

  let arguments = (ins eir_AnyTerm:$bin, Optional<eir_AnyTerm>:$size, OptionalAttr<I64Attr>:$unit);

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value bin, int64_t unit, Optional<Value> size",
    [{
      result.addOperands(bin);
      if (size.hasValue()) {
        result.addOperands(size.getValue());
      }
      auto termTy = builder.getType<TermType>();
      auto i1Ty = builder.getI1Type();
      auto i8Ty = builder.getIntegerType(8);
      result.addTypes({termTy, termTy, i1Ty});
      result.addAttribute("unit", builder.getIntegerAttr(i8Ty, unit));
    }]>
  ];
}

def eir_BinaryMatchIntegerOp : eir_BinaryMatchOp<"binary.match.integer", []> {
  let summary = "Binary pattern match integer values";

  let arguments = (ins eir_AnyTerm:$bin, Optional<eir_AnyTerm>:$size, BoolAttr:$is_signed, I8Attr:$endianness, OptionalAttr<I64Attr>:$unit);

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value bin, bool isSigned,"
    "Endianness::Type endianness, int64_t unit, Optional<Value> size",
    [{
      result.addOperands(bin);
      if (size.hasValue()) {
        result.addOperands(size.getValue());
      }
      auto termTy = builder.getType<TermType>();
      auto i1Ty = builder.getI1Type();
      auto i8Ty = builder.getIntegerType(8);
      result.addTypes({termTy, termTy, i1Ty});
      result.addAttribute("is_signed", builder.getBoolAttr(isSigned));
      result.addAttribute("endianness", builder.getIntegerAttr(i8Ty, (unsigned)endianness));
      result.addAttribute("unit", builder.getIntegerAttr(i8Ty, unit));
    }]>
  ];
}

def eir_BinaryMatchFloatOp : eir_BinaryMatchOp<"binary.match.float", []> {
  let summary = "Binary pattern match float values";

  let arguments = (ins eir_AnyTerm:$bin, Optional<eir_AnyTerm>:$size, I8Attr:$endianness, OptionalAttr<I64Attr>:$unit);

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value bin,"
    "Endianness::Type endianness, int64_t unit, Optional<Value> size",
    [{
      result.addOperands(bin);
      if (size.hasValue()) {
        result.addOperands(size.getValue());
      }
      auto termTy = builder.getType<TermType>();
      auto i1Ty = builder.getI1Type();
      auto i8Ty = builder.getIntegerType(8);
      result.addTypes({termTy, termTy, i1Ty});
      result.addAttribute("endianness", builder.getIntegerAttr(i8Ty, (unsigned)endianness));
      result.addAttribute("unit", builder.getIntegerAttr(i8Ty, unit));
    }]>
  ];
}

def eir_BinaryMatchUtf8Op : eir_BinaryMatchOp<"binary.match.utf8", []> {
  let summary = "Binary pattern match utf8 character values";

  let arguments = (ins eir_AnyTerm:$bin, Optional<eir_AnyTerm>:$size);

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value bin, Optional<Value> size",
    [{
      result.addOperands(bin);
      if (size.hasValue()) {
        result.addOperands(size.getValue());
      }
      auto termTy = builder.getType<TermType>();
      auto i1Ty = builder.getI1Type();
      result.addTypes({termTy, termTy, i1Ty});
    }]>
  ];
}

def eir_BinaryMatchUtf16Op : eir_BinaryMatchOp<"binary.match.utf16", []> {
  let summary = "Binary pattern match utf16 character values";

  let arguments = (ins eir_AnyTerm:$bin, Optional<eir_AnyTerm>:$size, I8Attr:$endianness);

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value bin,"
    "Endianness::Type endianness, Optional<Value> size",
    [{
      result.addOperands(bin);
      if (size.hasValue()) {
        result.addOperands(size.getValue());
      }
      auto termTy = builder.getType<TermType>();
      auto i1Ty = builder.getI1Type();
      auto i8Ty = builder.getIntegerType(8);
      result.addTypes({termTy, termTy, i1Ty});
      result.addAttribute("endianness", builder.getIntegerAttr(i8Ty, (unsigned)endianness));
    }]>
  ];
}

def eir_BinaryMatchUtf32Op : eir_BinaryMatchOp<"binary.match.utf32", []> {
  let summary = "Binary pattern match utf32 character values";

  let arguments = (ins eir_AnyTerm:$bin, Optional<eir_AnyTerm>:$size, I8Attr:$endianness);

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value bin,"
    "Endianness::Type endianness, Optional<Value> size",
    [{
      result.addOperands(bin);
      if (size.hasValue()) {
        result.addOperands(size.getValue());
      }
      auto termTy = builder.getType<TermType>();
      auto i1Ty = builder.getI1Type();
      auto i8Ty = builder.getIntegerType(8);
      result.addTypes({termTy, termTy, i1Ty});
      result.addAttribute("endianness", builder.getIntegerAttr(i8Ty, (unsigned)endianness));
    }]>
  ];
}

def eir_ReceiveStartOp : eir_Op<"receive.start", []> {
  let summary = "starts a receive operation";

  let arguments = (ins eir_AnyType:$timeout);
  let results = (outs eir_ReceiveRefType:$result);

  let verifier = ?;
  let hasCanonicalizer = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value timeout",
    [{
      result.addOperands(timeout);
      result.addTypes(builder.getType<ReceiveRefType>());
    }]>
  ];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def eir_ReceiveWaitOp : eir_Op<"receive.wait", []> {
  let summary = "Blocks until receipt of a new message";
  let arguments = (ins eir_ReceiveRefType:$recvRef);
  let results = (outs eir_ReceiveStatusType:$status);

  let verifier = ?;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value recvRef",
    [{
      result.addOperands(recvRef);
      result.addTypes(builder.getIntegerType(8));
    }]>
  ];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def eir_ReceiveMessageOp : eir_Op<"receive.message", []> {
  let summary = "Accesses a received message from the given context";
  let arguments = (ins eir_ReceiveRefType:$recvRef);
  let results = (outs eir_AnyType:$result);

  let verifier = ?;

  let builders = [
    OpBuilder<
    "OpBuilder &builder, OperationState &result, Value recvRef",
    [{
      result.addOperands(recvRef);
      result.addTypes(builder.getType<TermType>());
    }]>
  ];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def eir_ReceiveDoneOp : eir_Op<"receive.done", []> {
  let summary = "ends a receive operation";

  let arguments = (ins eir_ReceiveRefType:$recvRef);
  let results = (outs);

  let verifier = ?;

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}


#endif // EIR_OPS
