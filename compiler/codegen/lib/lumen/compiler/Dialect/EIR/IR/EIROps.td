#ifndef EIR_OPS
#define EIR_OPS

include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffects.td"

include "lumen/compiler/Dialect/EIR/IR/EIRBase.td"

def eir_FuncOp : eir_Op<"func", [
    IsolatedFromAbove,
    FunctionLike,
    CallableOpInterface,
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
      Builder *builder, OperationState &result, StringRef name,
      FunctionType type, ArrayRef<NamedAttribute> attrs = {},
      ArrayRef<NamedAttributeList> argAttrs = {}
    }]>,
  ];

  let verifier = ?;

  let extraClassDeclaration = [{

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
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }

    /// Hook for OpTrait::FunctionLike, returns the number of function results.
    /// Depends on the type attribute being correct as checked by verifyType.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present. This can check for preconditions of the
    /// getNumArguments hook not failing.
    LogicalResult verifyType();

    Region *getCallableRegion() { return &body(); }
    ArrayRef<Type> getCallableResults() {
      assert(!isExternal() && "invalid callable");
      return getType().getResults();
    }
  }];
}

def eir_ClosureOp : eir_Op<"closure", []> {
  let summary = "Creates a closure capture at the current point in the function";

  let arguments = (ins
    eir_FuncRefAttr:$callee,
    Variadic<eir_AnyType>:$operands
  );
  let results = (outs
    Variadic<eir_AnyType>:$results
  );

  let assemblyFormat = [{
    $callee `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, eir::Closure *closure, ValueRange operands = {}
    }], [{
      auto termTy = builder->getType<TermType>();
      auto i8Ty = builder->getIntegerType(8);
      auto i32Ty = builder->getIntegerType(32);
      auto callee = builder->getSymbolRefAttr(StringRef(closure->name));
      result.addAttribute("callee", callee);
      result.addAttribute("module", Attribute::getFromOpaquePointer(closure->module));
      result.addAttribute("arity", builder->getIntegerAttr(i8Ty, closure->arity));
      result.addAttribute("env_len", builder->getIntegerAttr(i8Ty, closure->envLen));
      result.addAttribute("index", builder->getIntegerAttr(i32Ty, closure->index));
      result.addAttribute("old_unique", builder->getIntegerAttr(i32Ty, closure->index));
      StringRef unique(&closure->unique[0], 16);
      result.addAttribute("unique", builder->getStringAttr(unique));
      result.addOperands(operands);
      result.addTypes(termTy);
    }]>
  ];

  let verifier = ?;

  let extraClassDeclaration = [{
    IntegerAttr arityAttr() { return getAttrOfType<IntegerAttr>("arity"); }
    AtomAttr module() { return getAttrOfType<AtomAttr>("module"); }
    IntegerAttr envLenAttr() { return getAttrOfType<IntegerAttr>("env_len"); }
    IntegerAttr indexAttr() { return getAttrOfType<IntegerAttr>("index"); }
    IntegerAttr oldUniqueAttr() { return getAttrOfType<IntegerAttr>("old_unique"); }
    StringAttr uniqueAttr() { return getAttrOfType<StringAttr>("unique"); }

    unsigned arity() { return arityAttr().getValue().getLimitedValue(); }
    unsigned envLen() { return envLenAttr().getValue().getLimitedValue(); }
    unsigned index() { return indexAttr().getValue().getLimitedValue(); }
    unsigned oldUnique() { return oldUniqueAttr().getValue().getLimitedValue(); }
    StringRef unique() { return uniqueAttr().getValue(); }

    bool isAnonymous() { return arityAttr() != nullptr; }
  }];
}

def eir_UnpackEnvOp : eir_Op<"closure.unpack", []> {
  let summary = "Unpacks a value from a closure environment";

  let arguments = (ins eir_BoxType:$env, Confined<I32Attr, [IntNonNegative]>:$index);
  let results = (outs eir_AnyType:$result);

  let assemblyFormat = [{
    $env `[` $index `]` attr-dict `:` functional-type(operands, results)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value env, unsigned index",
    [{
      result.addOperands(env);
      result.addTypes(builder->getType<TermType>());
      result.addAttribute("index", builder->getI32IntegerAttr(index));
    }]>
  ];

  let verifier = ?;

  let extraClassDeclaration = [{
    unsigned envIndex() { return getAttrOfType<IntegerAttr>("index").getValue().getLimitedValue(); }
  }];
}

//===----------------------------------------------------------------------===//
// Comparisons
//===----------------------------------------------------------------------===//

def eir_IsTypeOp : eir_Op<"is_type", []> {
  let summary = "A type checking primitive used in conditional branching.";

  let description = [{
    This operation is used to examine the type of a term value and return a
    boolean or boolean term indicating whether the provided type matches that
    of the value.
  }];

  let arguments = (ins eir_AnyType:$value);
  let results = (outs eir_BoolLike:$isMatch);

  let assemblyFormat = "`(` operands `)` attr-dict `:` functional-type(operands, $isMatch)";

  //let hasCanonicalizer = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, "
    "Value value, Type matchType, Type returnType",
    [{
      result.addOperands(value);
      result.addTypes(returnType);
      result.addAttribute("type", TypeAttr::get(matchType));
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, "
    "Value value, Type matchType",
    [{
      build(builder, result, value, matchType, builder->getType<BooleanType>());
    }]>
  ];

  let extraClassDeclaration = [{
    Type getValueType() {
      return getOperand().getType();
    }

    Type getMatchType() {
      Attribute attr = getAttr("type");
      TypeAttr typeAttr = attr.dyn_cast<TypeAttr>();
      return typeAttr.getValue();
    }

    Type getResultType() { return getResult().getType(); }
  }];
}

class eir_BinaryOperator<string mnemonic, Type left, Type right, Type result, list<OpTrait> traits = []>
    : eir_Op<mnemonic, !listconcat(traits, [eir_BinaryOperatorInterface])> {
  // Defines the name of the builtin associated with this comparison op
  string builtinSymbol = "__lumen_builtin_" # mnemonic;

  let summary = ?;
  let description = ?;

  let arguments = (ins left:$lhs, right:$rhs);
  let results = (outs result:$result);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, $result)";

  let verifier = ?;

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type resultTy, Value lhs, Value rhs,"
    "ArrayRef<NamedAttribute> attrs = {}",
    [{
      result.addOperands(ValueRange({lhs, rhs}));
      result.addTypes(resultTy);
      result.addAttributes(attrs);
    }]>
  ];

  let extraClassDeclaration = "static StringRef builtinSymbol() { return \"" # builtinSymbol # "\"; }";
}

class eir_LogicalOp<Type type, string mnemonic, list<OpTrait> traits = []> :
    eir_BinaryOperator<mnemonic, type, type, eir_BoolType, traits> {

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value lhs, Value rhs",
    [{ build(builder, result, builder->getType<BooleanType>(), lhs, rhs); }]>
  ];
}

def eir_LogicalAndOp : eir_LogicalOp<eir_AnyType, "logical.and", [Commutative]> {
  let summary = "logical AND";
  //let hasFolder = 1;
}

def eir_LogicalOrOp : eir_LogicalOp<eir_AnyType, "logical.or", [Commutative]> {
  let summary = "logical OR";
  //let hasFolder = 1;
}

class eir_UnaryComparisonOp<Type type, string mnemonic, list<OpTrait> traits = []> :
    eir_Op<mnemonic, traits> {
  let description = [{
    Compares a single operand against the specified predicate.
  }];

  let arguments = (ins
    type:$operand
  );
  let results = (outs
    eir_BoolType:$result
  );

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, $result)";
  let verifier = ?;
}

class eir_BinaryComparisonOp<Type type, string mnemonic, list<OpTrait> traits = []>
    : eir_BinaryOperator<mnemonic, type, type, eir_BoolType, traits> {

  let summary = "Compares two operands with the specified predicate.";

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, ArrayRef<Value> args",
    [{
      assert(args.size() == 2 && "expected binary operator to receive 2 arguments");
      Value lhs = args[0];
      Value rhs = args[1];
      build(builder, result, lhs, rhs);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, Value lhs, Value rhs, bool strict = false",
    [{
      auto boolType = builder->getType<eir::BooleanType>();
      result.addAttribute("strict", builder->getBoolAttr(strict));
      build(builder, result, boolType, lhs, rhs);
    }]>
  ];
}

def eir_CmpEqOp :
    eir_BinaryComparisonOp<eir_AnyType, "cmp.eq", [Commutative]> {
  let summary = [{term equality comparison operation}];
  //let hasFolder = 1;
}

def eir_CmpNeqOp :
    eir_BinaryComparisonOp<eir_AnyType, "cmp.neq", [Commutative]> {
  let summary = [{term inequality comparison operation}];
  //let hasFolder = 1;
}

def eir_CmpLtOp :
    eir_BinaryComparisonOp<eir_AnyType, "cmp.lt"> {
  let summary = [{term less-than comparison operation}];
  //let hasFolder = 1;
}

def eir_CmpLteOp :
    eir_BinaryComparisonOp<eir_AnyType, "cmp.lte"> {
  let summary = [{term less-than-or-equal comparison operation}];
  //let hasFolder = 1;
}

def eir_CmpGtOp :
    eir_BinaryComparisonOp<eir_AnyType, "cmp.gt"> {
  let summary = [{term greater-than comparison operation}];
  //let hasFolder = 1;
}

def eir_CmpGteOp :
    eir_BinaryComparisonOp<eir_AnyType, "cmp.gte"> {
  let summary = [{term greater-than-or-equal comparison operation}];
  //let hasFolder = 1;
}

class eir_BinaryArithmeticOp<Type type, string mnemonic, list<OpTrait> traits = []>
    : eir_BinaryOperator<mnemonic, type, type, type, traits> {
  let summary = "Peform the specified arithmetic operation over two operands";

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, ArrayRef<Value> args",
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
    eir_BinaryArithmeticOp<eir_AnyType, "math.add", [Commutative]> {
  let summary = "Addition operator";
}

def eir_SubOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.sub"> {
  let summary = "Subraction operator";
}

def eir_MulOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.mul", [Commutative]> {
  let summary = "Multiplication operator";
}

def eir_DivOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.div"> {
  let summary = "Integer division operator";
}

def eir_FDivOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.fdiv"> {
  let summary = "Floating-point division operator";
}

def eir_RemOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.rem"> {
  let summary = "Integer remainder operator";
}

def eir_BslOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.bsl"> {
  let summary = "Bitwise shift-left operator";
}

def eir_BsrOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.bsr"> {
  let summary = "Bitwise shift-right operator";
}

def eir_BandOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.band"> {
  let summary = "Bitwise AND operator";
}

def eir_BorOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.bor"> {
  let summary = "Bitwise OR operator";
}

def eir_BxorOp :
    eir_BinaryArithmeticOp<eir_AnyType, "math.bxor"> {
  let summary = "Bitwise XOR operator";
}

//===----------------------------------------------------------------------===//
// Control Flow
//===----------------------------------------------------------------------===//

def eir_BranchOp : eir_Op<"br",
    [DeclareOpInterfaceMethods<BranchOpInterface>, NoSideEffect, Terminator]> {
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

  let arguments = (ins Variadic<eir_AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);

  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, Block *dest,
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
    void eraseOperand(unsigned index) { return eraseSuccessorOperand(0, index); }
  }];

  let hasCanonicalizer = 1;
  let assemblyFormat = [{
      $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}

def eir_CondBranchOp : eir_Op<"cond_br",
  [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>,
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
    eir_BoolType:$condition,
    Variadic<eir_AnyType>:$trueDestOperands,
    Variadic<eir_AnyType>:$falseDestOperands
  );

  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, Value condition,
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

    OperandRange getTrueOperands() { return trueDestOperands(); }

    unsigned getNumTrueOperands()  { return getTrueOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      eraseSuccessorOperand(trueIndex, index);
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

    OperandRange getFalseOperands() { return falseDestOperands(); }

    unsigned getNumFalseOperands() { return getFalseOperands().size(); }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      eraseSuccessorOperand(falseIndex, index);
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
    $condition `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict
  }];
}

class eir_CallBaseOp<string mnemonic, list<OpTrait> traits = []> :
    eir_Op<mnemonic, !listconcat(traits, [CallOpInterface])> {
  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }

    /// Get the argument operands to the called function.
    OperandRange getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<FlatSymbolRefAttr>("callee");
    }
  }];
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

    OperandRange getOkOperands() { return okDestOperands(); }

    unsigned getNumOkOperands()  { return getOkOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseOkOperand(unsigned index)  {
      eraseSuccessorOperand(okIndex, index);
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

    OperandRange getErrOperands() { return errDestOperands(); }

    unsigned getNumErrOperands() { return getErrOperands().size(); }

    /// Erase the operand at 'index' from the false operand list.
    void eraseErrOperand(unsigned index) {
      eraseSuccessorOperand(errIndex, index);
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

def eir_CallOp : eir_CallBaseOp<"call"> {
  let summary = [{call operation}];
  let description = [{
    Calls a function with the given arguments.
  }];

  let arguments = (ins
    eir_FuncRefAttr:$callee,
    Variadic<eir_AnyType>:$operands
  );
  let results = (outs
    Variadic<eir_AnyType>:$results
  );

  let assemblyFormat = [{
    $callee `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, eir::FuncOp callee,
      ValueRange operands = {}
    }], [{
      result.addOperands(operands);
      result.addAttribute("callee", builder->getSymbolRefAttr(callee));
      result.addTypes(callee.getType().getResults());
    }]>,
    OpBuilder<[{
      Builder *builder, OperationState &result, FlatSymbolRefAttr callee,
      ArrayRef<Type> resultTypes, ValueRange operands = {}
    }], [{
      result.addOperands(operands);
      result.addAttribute("callee", callee);
      result.addTypes(resultTypes);
    }]>,
    OpBuilder<[{
      Builder *builder, OperationState &result, StringRef callee,
      ArrayRef<Type> resultTypes, ValueRange operands = {}
    }], [{
      build(builder, result, builder->getSymbolRefAttr(callee),
            resultTypes, operands);
    }]>,
  ];

  let verifier = [{ /*TODO*/return success(); }];

  //let hasCanonicalizer = 1;
}

def CallIndirectOp : eir_Op<"call_indirect", [CallOpInterface]> {
  let summary = "indirect call operation";
  let description = [{
    The "call_indirect" operation represents an indirect call to a value of
    function type.  Functions are first class types in MLIR, and may be passed
    as arguments and merged together with block arguments.  The operands
    and result types of the call must match the specified function type.

      %3 = eir.call_indirect %2(%0, %1) : (f32, f32) -> f32
  }];

  let arguments = (ins FunctionType:$callee, Variadic<eir_AnyType>:$operands);
  let results = (outs Variadic<eir_AnyType>:$results);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value callee,"
    "ValueRange operands = {}", [{
      result.operands.push_back(callee);
      result.addOperands(operands);
      FunctionType fnTy = callee.getType().cast<FunctionType>();
      result.addTypes(fnTy.getResults());
  }]>];

  let extraClassDeclaration = [{
    Value getCallee() { return getOperand(0); }

    /// Get the argument operands to the called function.
    OperandRange getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }
  }];

  let verifier = ?;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def eir_InvokeOp : eir_InvokeBaseOp<"invoke", [CallOpInterface]> {
  let summary = "call operation for targets that may raise exceptions";
  let description = [{
    Calls a function with the given arguments, uses the provided ok/err blocks
    to handle the flow of control upon return of the callee, based on whether
    an exception was raised or not.
  }];

  let arguments = (ins
    eir_FuncRefAttr:$callee,
    Variadic<eir_AnyType>:$operands,
    Variadic<eir_AnyType>:$okDestOperands,
    Variadic<eir_AnyType>:$errDestOperands
  );
  let results = (outs
    Variadic<eir_AnyType>:$results
  );

  let assemblyFormat = [{
    $callee (`(` $operands^ `:` type($operands) `)`)? `to`
      $okDest (`(` $okDestOperands^ `:` type($okDestOperands) `)`)? `unwind`
      $errDest (`(` $errDestOperands^ `:` type($errDestOperands) `)`)?
      attr-dict `:` type(results)
  }];

  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, eir::FuncOp callee,
      ValueRange operands,
      Block *okDest, ValueRange okDestOperands,
      Block *errDest, ValueRange errDestOperands
    }], [{
      build(builder, result, callee.getCallableResults(), builder->getSymbolRefAttr(callee),
            operands, okDestOperands, errDestOperands, okDest, errDest);
    }]>,
    OpBuilder<[{
      Builder *builder, OperationState &result, FlatSymbolRefAttr callee,
      ArrayRef<Type> resultTypes, ValueRange operands,
      Block *okDest, ValueRange okDestOperands,
      Block *errDest, ValueRange errDestOperands
    }], [{
      build(builder, result, resultTypes, callee,
            operands, okDestOperands, errDestOperands, okDest, errDest);
    }]>,
    OpBuilder<[{
      Builder *builder, OperationState &result, StringRef callee,
      ArrayRef<Type> resultTypes, ValueRange operands,
      Block *okDest, ValueRange okOperands,
      Block *errDest, ValueRange errOperands
    }], [{
      build(builder, result, resultTypes, builder->getSymbolRefAttr(callee),
            operands, okOperands, errOperands, okDest, errDest);
    }]>,
  ];

  let verifier = [{ /*TODO*/return success(); }];

  //let hasCanonicalizer = 1;
  let extraExtraClassDeclaration = [{
    StringRef getCallee() { return callee(); }

    /// Get the argument operands to the called function.
    OperandRange getArgOperands() { return operands(); }

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
  let results = (outs eir_AnyType:$kind, eir_AnyType:$reason, eir_AnyType:$trace);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let verifier = ?;

  //let hasCanonicalizer = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, ValueRange catchClauses", [{
      Type termTy = builder->getType<TermType>();
      build(builder, result, ArrayRef<Type>{termTy, termTy, termTy}, catchClauses);
  }]>];
}

def ApplyOp : eir_Op<"apply", []> {
  let summary = "dynamic call operation";
  let description = [{
    The "apply" operation represents an dynamic call to an MFA.

    The first two operands are the module, and function; the arity
    is inferred by the number of arguments passed.

      %0 = eir.constant.atom { value = "erlang" }
      %1 = eir.constant.atom { value = "+" }
      %4 = eir.apply [%0, %1](%2, %3) : (f32, f32) -> f32
  }];

  let arguments = (ins eir_AnyType:$module, eir_AnyType:$fun, Variadic<eir_AnyType>:$operands);
  let results = (outs eir_AnyType:$result);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value module, Value fun,"
    "ValueRange operands = {}", [{
      result.operands.push_back(module);
      result.operands.push_back(fun);
      result.addOperands(operands);
      result.addTypes(builder->getType<TermType>());
  }]>];

  let extraClassDeclaration = [{
    Value getTargetModule() { return getOperand(0); }
    Value getTargetFun() { return getOperand(1); }

    /// Get the argument operands to the called function.
    OperandRange getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin() + 2; }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];

  let verifier = ?;

  let assemblyFormat = "`[` $module `:` type($module) `,` $fun `:` type($fun) `]` `(` $operands `)` attr-dict `:` functional-type($operands, $result)";
}

def CallClosureOp : eir_Op<"call_closure", []> {
  let summary = "operation to call a closure";
  let description = [{
      %3 = eir.call_closure %2(%0, %1) : (f32, f32) -> f32
  }];

  let arguments = (ins eir_BoxType:$callee, Variadic<eir_AnyType>:$operands);
  let results = (outs eir_AnyType:$result);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value callee,"
    "ValueRange operands = {}", [{
      result.operands.push_back(callee);
      result.addOperands(operands);
      FunctionType fnTy = callee.getType().cast<FunctionType>();
      auto closureTy = builder->getType<ClosureType>();
      result.addTypes(fnTy.getResults());
  }]>];

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    OperandRange getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];

  let verifier = ?;
  //let hasCanonicalizer = 1;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def eir_InvokeClosureOp : eir_InvokeBaseOp<"invoke_closure", []> {
  let summary = "call operation for closures that may raise exceptions";
  let description = [{
    Calls a closure with the given arguments, uses the provided ok/err blocks
    to handle the flow of control upon return of the callee, based on whether
    an exception was raised or not.
  }];

  let arguments = (ins
    eir_BoxType:$callee,
    Variadic<eir_AnyType>:$operands,
    Variadic<eir_AnyType>:$okDestOperands,
    Variadic<eir_AnyType>:$errDestOperands
  );
  let results = (outs eir_AnyType:$result);

  let assemblyFormat = [{
    $callee `:` type($callee) (`(` $operands^ `:` type($operands) `)`)? `to`
      $okDest (`(` $okDestOperands^ `:` type($okDestOperands) `)`)? `unwind`
      $errDest (`(` $errDestOperands^ `:` type($errDestOperands) `)`)?
      attr-dict `:` type(results)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, Value callee,
      ValueRange operands,
      Block *okDest, ValueRange okDestOperands,
      Block *errDest, ValueRange errDestOperands
    }], [{
      result.operands.push_back(callee);
      result.addOperands(operands);
      result.addOperands(okDestOperands);
      result.addOperands(errDestOperands);
      FunctionType fnTy = callee.getType().cast<FunctionType>();
      auto closureTy = builder->getType<ClosureType>();
      result.addTypes(fnTy.getResults());
      result.addSuccessors({okDest, errDest});
    }]>,
  ];

  let verifier = [{ /*TODO*/return success(); }];

  //let hasCanonicalizer = 1;

  let extraExtraClassDeclaration = [{
    /// Get the argument operands to the called function.
    OperandRange getArgOperands() { return operands(); }

    unsigned getNumArgOperands() { return getArgOperands().size(); }
  }];
}

def eir_ReturnOp : eir_Op<"return", [NoSideEffect, Terminator]> {
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

  let builders = [OpBuilder<
    "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def eir_YieldOp : eir_Op<"yield", [ YieldPoint]> {
  let summary = "scheduler yield operation";
  let description = [{
    Yields execution back to the scheduler
  }];

  let verifier = ?;
  let assemblyFormat = [{ attr-dict }];
}

def eir_YieldCheckOp : eir_Op<"yield.check",
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>,
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
      Builder *builder, OperationState &result,
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

    OperandRange getTrueOperands() { return trueDestOperands(); }

    unsigned getNumTrueOperands()  { return getTrueOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      eraseSuccessorOperand(trueIndex, index);
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

    OperandRange getFalseOperands() { return falseDestOperands(); }

    unsigned getNumFalseOperands() { return getFalseOperands().size(); }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      eraseSuccessorOperand(falseIndex, index);
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
    eir_AnyType:$trace
  );

  let verifier = ?;

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

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result",
    [{
      result.addAttribute("increment", builder->getI32IntegerAttr(1));
    }]>
  ];

  let verifier = ?;

  let assemblyFormat = [{ attr-dict }];
}

def eir_CastOp : eir_Op<"cast", []> {

  let arguments = (ins eir_AnyType:$input);
  let results = (outs eir_AnyType:$output);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, Value source, Type destType",
    [{
      result.addOperands(source);
      result.addAttribute("from", TypeAttr::get(source.getType()));
      result.addAttribute("to", TypeAttr::get(destType));
      result.addTypes(destType);
    }]>
  ];

  let assemblyFormat = [{
    $input `:` type($input) `to` type($output) attr-dict
  }];

  //let hasFolder = 1;
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

  let arguments = (ins Variadic<eir_AnyType>:$arity,
                   Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment);
  let results = (outs eir_BoxType);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, BoxType boxType, "
              "IntegerAttr alignment = IntegerAttr()",
    [{
       result.addAttribute("type", TypeAttr::get(boxType.getBoxedType()));
       result.types.push_back(boxType);
       if (alignment)
         result.addAttribute(getAlignmentAttrName(), alignment);
    }]>,
    OpBuilder<"Builder *builder, OperationState &result, BoxType boxType, "
              "ArrayRef<Value> operands, IntegerAttr alignment = IntegerAttr()",
    [{
       result.addAttribute("type", TypeAttr::get(boxType.getBoxedType()));
       result.types.push_back(boxType);
       if (alignment)
         result.addAttribute(getAlignmentAttrName(), alignment);
       result.addOperands(operands);
    }]>
  ];

  let extraClassDeclaration = [{
    static StringRef getAlignmentAttrName() { return "alignment"; }

    OpaqueTermType getAllocType() {
        return getOperation()->getAttrOfType<TypeAttr>("type").getValue().cast<OpaqueTermType>();
    }
  }];

  let hasCanonicalizer = 1;
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
    "Builder *builder, OperationState &result, Value refVal",
    [{
      result.addOperands(refVal);
      if (auto boxed = refVal.getType().dyn_cast_or_null<eir::BoxType>()) {
        result.addTypes(boxed.getBoxedType());
      } else if (auto ptr = refVal.getType().dyn_cast_or_null<eir::RefType>()) {
        result.addTypes(ptr.getInnerType());
      } else {
        result.addTypes(refVal.getType());
      }
    }]
  >];

  let verifier = [{
    auto refVal = ref();
    if (!refVal) {
        return emitError("invalid value: referenced value must not be null");
    }
    if (refVal.getType().isa<eir::BoxType>()) {
        return success();
    }
    if (refVal.getType().isa<eir::RefType>()) {
        return success();
    }
    return emitError("invalid value: expected referenced type to be a box or ref type");
  }];

  let assemblyFormat = [{
    `(` $ref `)` attr-dict `:` functional-type($ref, $out)
  }];
}

def eir_GetElementPtrOp : eir_Op<"getelementptr", []> {
  let summary = "An aggregate access primitive used to calculate memory addresses to an element of an aggregate type";
  let description = [{
    An aggregate access primitive used to calculate memory addresses to an element of an aggregate type
  }];

  let arguments = (ins eir_BoxType:$base, Variadic<eir_AnyType>:$indices);
  let results = (outs eir_PointerLike:$ref);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value boxedVal, int64_t index",
    [{
      auto boxedTy = boxedVal.getType();
      Type elementTy;
      if (auto boxTy = boxedTy.dyn_cast_or_null<eir::BoxType>()) {
        elementTy = boxTy.getBoxedType();
      } else {
        assert(false && "expected element type to be a boxed term type");
      }
      result.addAttribute("pointee", TypeAttr::get(elementTy));
      auto termTy = builder->getType<eir::TermType>().cast<OpaqueTermType>();
      if (elementTy.isa<ConsType>()) {
        result.addAttribute("element", TypeAttr::get(termTy));
        auto refTy = builder->getType<eir::RefType>(termTy);
        result.addTypes(refTy);
      } else if (auto tupleTy = elementTy.dyn_cast_or_null<TupleType>()) {
        if (tupleTy.hasDynamicShape()) {
          result.addAttribute("element", TypeAttr::get(termTy));
          auto refTy = builder->getType<eir::RefType>(termTy);
          result.addTypes(refTy);
        } else if (tupleTy.getArity() > index) {
          auto innerTy = tupleTy.getElementType(index).cast<OpaqueTermType>();
          result.addAttribute("element", TypeAttr::get(innerTy));
          auto refTy = builder->getType<eir::RefType>(innerTy);
          result.addTypes(refTy);
        } else {
          result.addAttribute("element", TypeAttr::get(termTy));
          auto refTy = builder->getType<eir::RefType>(termTy);
          result.addTypes(refTy);
        }
      } else {
        result.addAttribute("element", TypeAttr::get(termTy));
        auto refTy = builder->getType<eir::RefType>(termTy);
        result.addTypes(refTy);
      }
      result.addOperands(boxedVal);
      auto indexTy = builder->getIndexType();
      result.addAttribute("index", builder->getIntegerAttr(indexTy, index));
    }]>
  ];

  let verifier = ?;

  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    static StringRef getIndexAttrName() { return "index"; }

    Type getPointeeType() { return getAttrOfType<TypeAttr>("pointee").getValue(); }
    Type getElementType() { return getAttrOfType<TypeAttr>("element").getValue(); }
    Type getType() { return getResult().getType().cast<RefType>().getInnerType(); }

    uint64_t getIndex() {
        return getAttrOfType<IntegerAttr>("index").getValue().getLimitedValue();
    }
  }];

}

def eir_PrintOp : eir_Op<"print"> {
  let summary = "intrinsic printing operation";
  let description = [{
    Prints to standard output using the provided format string and operands.
  }];

  let arguments = (ins
    Variadic<eir_AnyType>:$operands
  );
  let results = (outs eir_AnyTerm:$result);

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, ArrayRef<Value> args",
    [{
      Value arg = args.front();
      Type termTy = builder->getType<TermType>();
      build(builder, result, termTy, arg);
    }]>
  ];

  let verifier = ?;

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

class eir_ConstantOp<Type type, string mnemonic> : eir_Op<mnemonic, [ConstantLike, NoSideEffect]> {

  let skipDefaultBuilders = 1;
  let summary = "Constructs a constant term value";

  let arguments = (ins AnyAttr:$value);
  let results = (outs type);

  let parser = [{ return parseConstantOp(parser, result); }];
  let printer = [{ printConstantOp(p, *this); }];
  let verifier = [{ return verifyConstantOp(*this); }];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
  }];

  let hasFolder = 1;
}

def eir_ConstantFloatOp : eir_ConstantOp<eir_FloatLike, "constant.float"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, APFloat &value",
    [{
      auto type = builder->getType<eir::FloatType>();
      auto attr = builder->getFloatAttr(type, value);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantIntOp : eir_ConstantOp<eir_FixnumLike, "constant.int"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, int64_t value",
    [{
      auto type = builder->getIntegerType(64);
      auto n = static_cast<uint64_t>(value);
      APInt ap(64, n, /*isSigned=*/true);
      auto attr = builder->getIntegerAttr(type, ap);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantBigIntOp : eir_ConstantOp<eir_BigIntType, "constant.bigint"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, APInt &value",
    [{
      auto type = builder->getType<BigIntType>();
      auto attr = builder->getIntegerAttr(type, value);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantAtomOp : eir_ConstantOp<eir_AtomType, "constant.atom"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, bool truth",
    [{
      auto id = APInt(64, truth ? 1 : 0, /*signed=*/false);
      if (truth) {
        build(builder, result, id, "true");
      } else {
        build(builder, result, id, "false");
      }
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, APInt id, StringRef name",
    [{
      auto type = builder->getType<AtomType>();
      auto attr = AtomAttr::get(builder->getContext(), id, name);
      build(builder, result, type, attr);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, APInt id, StringRef name",
    [{
      auto attr = AtomAttr::get(builder->getContext(), id, name);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantBinaryOp : eir_ConstantOp<eir_BinaryType, "constant.binary"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result,"
    "StringRef value, uint64_t header, uint64_t flags",
    [{
      auto type = builder->getType<eir::BinaryType>();
      auto attr = BinaryAttr::get(type, value, header, flags);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantNilOp : eir_ConstantOp<eir_NilType, "constant.nil"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, Type type",
    [{
      auto attr = TypeAttr::get(type);
      result.addAttribute("value", attr);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result",
    [{
      auto type = builder->getType<NilType>();
      build(builder, result, type);
    }]>
  ];
}

def eir_ConstantNoneOp : eir_ConstantOp<eir_NoneType, "constant.none"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result",
    [{
      auto type = builder->getType<eir::NoneType>();
      auto attr = TypeAttr::get(type);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantTupleOp : eir_ConstantOp<eir_TupleType, "constant.tuple"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, ArrayRef<Attribute> elements",
    [{
      std::vector<Type> types;
      types.reserve(elements.size());
      for (auto element : elements) {
        types.push_back(element.getType());
      }
      auto type = builder->getType<eir::TupleType>(ArrayRef(types));
      auto attr = SeqAttr::get(type, elements);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantConsOp : eir_ConstantOp<eir_BoxType, "constant.cons"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, ArrayRef<Attribute> elements",
    [{
      assert(elements.size() > 0 && "cons requires at least one value");
      auto type = builder->getType<ConsType>();
      result.addAttribute("value", SeqAttr::get(type, elements));
      result.addTypes(BoxType::get(type));
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, Attribute head, Attribute tail",
    [{
      auto type = builder->getType<ConsType>();
      auto attr = SeqAttr::get(type, ArrayRef<Attribute>{head, tail});
      result.addAttribute("value", attr);
      result.addTypes(BoxType::get(type));
    }]>
  ];
}

def eir_ConstantListOp : eir_ConstantOp<eir_BoxType, "constant.list"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, ArrayRef<Attribute> elements",
    [{
      auto type = builder->getType<ConsType>();
      auto attr = SeqAttr::get(type, elements);
      build(builder, result, BoxType::get(type), attr);
    }]>
  ];
}

def eir_ConstantMapOp : eir_ConstantOp<eir_MapType, "constant.map"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, ArrayRef<Attribute> elements",
    [{
      auto type = builder->getType<MapType>();
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

  let arguments = (ins eir_AnyType:$head, eir_AnyType:$tail);
  let results = (outs eir_BoxType:$out);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, Value head, Value tail",
    [{
      result.addOperands(head);
      result.addOperands(tail);
      result.addTypes(BoxType::get(builder->getType<ConsType>()));
    }]>
  ];

  let verifier = ?;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, $out)
  }];

  let extraClassDeclaration = [{
      bool useAlloca() { return getAttrOfType<BoolAttr>("alloca").getValue(); }
  }];
}

def eir_TupleOp : eir_Op<"tuple"> {
  let summary = "Tuple constructor";
  let description = [{
    Tuple construction primitive. Constructs a new tuple term from a list of elements.
  }];

  let arguments = (ins Variadic<eir_AnyType>:$elements);
  let results = (outs eir_BoxType:$out);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, ArrayRef<Value> elements",
    [{
      SmallVector<Type, 1> elementTypes;
      for (auto val : elements) {
        elementTypes.push_back(val.getType());
      }
      result.addOperands(elements);
      auto tupleType = builder->getType<eir::TupleType>(elementTypes);
      result.addTypes(BoxType::get(tupleType));
      result.addAttribute("alloca", builder->getBoolAttr(false));
    }]>
  ];

  let verifier = ?;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, $out)
  }];

  let extraClassDeclaration = [{
      bool useAlloca() { return getAttrOfType<BoolAttr>("alloca").getValue(); }
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
  let results = (outs eir_AnyTerm:$capture);

  let verifier = ?;

  let assemblyFormat = [{ attr-dict `:` type($capture) }];
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

  let arguments = (ins);
  let results = (outs eir_AnyTerm:$trace);

  let verifier = ?;

  let assemblyFormat = [{ attr-dict `:` type($trace) }];
}

def eir_ConstructMapOp : eir_Op<"map.new"> {
  let summary = "Map constructor";
  let description = [{
    Map construction primitive. Creates a new map from a list of key/value pairs.
  }];

  let arguments = (ins Variadic<eir_AnyType>:$args);
  let results = (outs eir_BoxType:$out, eir_AnyType:$err);

  let verifier = ?;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, ArrayRef<eir::MapEntry> entries",
    [{
      for (auto &entry : entries) {
        result.addOperands(Value::getFromOpaquePointer(entry.key));
        result.addOperands(Value::getFromOpaquePointer(entry.value));
      }
      result.addTypes(BoxType::get(builder->getType<MapType>()));
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

        %0, %err = eir.map_insert %map, [(%k1 : !eir.fixnum, %v1 : !eir.fixnum)]
  }];

  let arguments = (ins eir_MapType:$map, Variadic<eir_AnyType>:$args);
  let results = (outs eir_MapType:$newMap, eir_AnyType:$err);

  let verifier = ?;

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value map, ValueRange pairs",
    [{
      auto mapType = builder->getType<MapType>();
      auto termType = builder->getType<TermType>();
      result.addTypes(mapType);
      result.addTypes(termType);
      result.addOperands(map);
      result.addOperands(pairs);
    }]>
  ];

  let assemblyFormat = [{
    $map `(` $args `)` attr-dict `:` functional-type(operands, results)
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

        %0, %err = eir.map_update %map, [(%k1 : !eir.fixnum, %v1 : !eir.fixnum)]
  }];

  let arguments = (ins eir_MapType:$map, Variadic<eir_AnyType>:$args);
  let results = (outs eir_MapType:$newMap, eir_AnyType:$err);

  let verifier = ?;

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value map, ValueRange pairs",
    [{
      auto mapType = builder->getType<MapType>();
      auto termType = builder->getType<TermType>();
      result.addTypes(mapType);
      result.addTypes(termType);
      result.addOperands(map);
      result.addOperands(pairs);
    }]>
  ];

  let assemblyFormat = [{
    $map `(` $args `)` attr-dict `:` functional-type(operands, results)
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
  let results = (outs eir_AnyType:$bin);

  let verifier = ?;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
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

  let arguments = (ins eir_AnyType:$head, eir_AnyType:$tail, Variadic<eir_AnyType>:$size);
  let results = (outs eir_AnyType:$bin, I1:$status);

  let verifier = ?;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value head, Value tail, Value size, NamedAttributeList attrs",
    [{
      if (size) {
        result.addOperands({head, tail, size});
      } else {
        result.addOperands({head, tail});
      }
      auto binTy = head.getType();
      auto i1Ty = builder->getI1Type();
      result.addTypes({binTy, i1Ty});
      result.addAttributes(attrs.getAttrs());
    }]>
  ];

  let verifier = ?;
}

def eir_ReceiveWaitOp : eir_Op<"receive.wait",
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>, Terminator]> {
  let summary = "Blocks until receipt of a new message";
  let arguments = (ins
    eir_ReceiveRefType:$recvRef,
    Variadic<eir_AnyType>:$timeoutDestOperands,
    Variadic<eir_AnyType>:$checkDestOperands
  );
  let results = (outs);

  let successors = (successor AnySuccessor:$timeoutDest, AnySuccessor:$checkDest);

  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, Value receiveRef,
      Block *timeoutDest, ValueRange timeoutOperands,
      Block *checkDest, ValueRange checkOperands
    }], [{
      build(builder, result, receiveRef, timeoutOperands, checkOperands, timeoutDest, checkDest);
    }]>,
  ];

  // Fully verified by traits
  let verifier = ?;

  let extraClassDeclaration = [{
    /// These are the indices into the dests list.
    enum { timeoutIndex = 0, checkIndex = 1 };

    /// The receive_ref operand is the first operand in the list.
    Value getReceiveRef() { return getOperand(0); }

    /// Return the destination if the operation times out
    Block *getTimeoutDest() {
      return getSuccessor(timeoutIndex);
    }

    /// Return the destination if the operation finds a message
    Block *getCheckDest() {
      return getSuccessor(checkIndex);
    }

    /// Accessors for operands to the 'true' destination.
    Value getTimeoutOperand(unsigned idx) {
      assert(idx < getNumTimeoutOperands());
      return getOperand(getTimeoutDestOperandIndex() + idx);
    }

    void setTimeoutOperand(unsigned idx, Value value) {
      assert(idx < getNumTimeoutOperands());
      setOperand(getTimeoutDestOperandIndex() + idx, value);
    }

    OperandRange getTimeoutOperands() { return timeoutDestOperands(); }

    unsigned getNumTimeoutOperands()  { return getTimeoutOperands().size(); }

    /// Erase the operand at 'index' from the timeout operand list.
    void eraseTimeoutOperand(unsigned index)  {
      eraseSuccessorOperand(timeoutIndex, index);
    }

    /// Accessors for operands to the 'check' destination.
    Value getCheckOperand(unsigned idx) {
      assert(idx < getNumCheckOperands());
      return getOperand(getCheckDestOperandIndex() + idx);
    }
    void setCheckOperand(unsigned idx, Value value) {
      assert(idx < getNumCheckOperands());
      setOperand(getCheckDestOperandIndex() + idx, value);
    }

    OperandRange getCheckOperands() { return checkDestOperands(); }

    unsigned getNumCheckOperands() { return getCheckOperands().size(); }

    /// Erase the operand at 'index' from the check operand list.
    void eraseCheckOperand(unsigned index) {
      eraseSuccessorOperand(checkIndex, index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTimeoutDestOperandIndex() { return 1; }

    /// Get the index of the first false destination operand.
    unsigned getCheckDestOperandIndex() {
      return getTimeoutDestOperandIndex() + getNumTimeoutOperands();
    }
  }];

  //let hasCanonicalizer = 1;
  let assemblyFormat = [{
    $recvRef `:` type($recvRef) `,`
    $timeoutDest (`(` $timeoutDestOperands^ `:` type($timeoutDestOperands) `)`)? `,`
    $checkDest (`(` $checkDestOperands^ `:` type($checkDestOperands) `)`)?
    attr-dict
  }];
}

#endif // EIR_OPS
