#ifndef EIR_OPS
#define EIR_OPS

include "mlir/Analysis/CallInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "lumen/compiler/Dialect/EIR/IR/EIRBase.td"

def eir_FuncOp : eir_Op<"func", [
    IsolatedFromAbove,
    HasParent<"ModuleOp">,
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

  let verifier = [{ return success(); }];

  let extraClassDeclaration = [{
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

def eir_ExportOp : eir_Op<"export", [
    HasParent<"ModuleOp">,
  ]> {
  let summary = [{exports a function from the module}];
  let description = [{
    Specifies an exported function with an externally-visible alias. Multiple
    exports can reference the same internal functions.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$function_ref,
    StrAttr:$export_name
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, eir::FuncOp functionRef,
      StringRef exportName = "", ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      Builder *builder, OperationState &result, FlatSymbolRefAttr functionRef,
      StringRef exportName = "", ArrayRef<NamedAttribute> attrs = {}
    }]>,
  ];

  let verifier = [{ return success(); }];
}

def eir_ImportOp : eir_Op<"import", [
    Symbol,
    NativeOpTrait<"FunctionLike">,
    CallableOpInterface,
  ]> {
  let summary = [{imports a function from an external module}];
  let description = [{
    Specifies a function that should be imported from either the runtime or
    an external module.
  }];

  let arguments = (ins);
  let regions = (region AnyRegion:$body);

  let verifier = [{ return success(); }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, StringRef name,
      FunctionType type, ArrayRef<NamedAttribute> attrs = {},
      ArrayRef<NamedAttributeList> argAttrs = {}
    }]>,
  ];

  let extraClassDeclaration = [{
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

    Region *getCallableRegion() { return nullptr; }
    ArrayRef<Type> getCallableResults() {
      return getType().getResults();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Comparisons
//===----------------------------------------------------------------------===//

def eir_IsTypeOp : eir_Op<"is_type", [NoSideEffect]> {
  let summary = "A type checking primitive used in conditional branching.";

  let description = [{
    This operation is used to examine the type of a term value and return a
    boolean or boolean term indicating whether the provided type matches that
    of the value.
  }];

  let arguments = (ins eir_AnyTerm:$value);
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

class eir_LogicalOp<Type type, string mnemonic, list<OpTrait> traits = []> :
      eir_Op<mnemonic, !listconcat(traits, [AllTypesMatch<["lhs", "rhs"]>])> {

  let arguments = (ins
      type:$lhs,
      type:$rhs
  );
  let results = (outs
    eir_BoolType:$result
  );

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, $result)";
  let verifier = [{ return mlir::success(); }];

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value lhs, Value rhs",
    [{
      result.addOperands(lhs);
      result.addOperands(rhs);
      auto boolType = builder->getType<BooleanType>();
      result.addTypes(boolType);
    }]>
  ];
}

def eir_LogicalAndOp : eir_LogicalOp<eir_AnyTerm, "logical.and", [Commutative]> {
  let summary = "logical AND";
  //let hasFolder = 1;
}

def eir_LogicalOrOp : eir_LogicalOp<eir_AnyTerm, "logical.or", [Commutative]> {
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
  let verifier = [{ return mlir::success(); }];
}

class eir_BinaryComparisonOp<Type type, string mnemonic, list<OpTrait> traits = []> : eir_Op<mnemonic, traits> {
  let description = [{
    Compares two operands with the specified predicate.
  }];

  let arguments = (ins
    type:$lhs,
    type:$rhs
  );
  let results = (outs
    eir_BoolType:$result
  );

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, $result)";
  let verifier = [{ return mlir::success(); }];

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value lhs, Value rhs",
    [{
      result.addOperands(lhs);
      result.addOperands(rhs);
      auto boolType = builder->getType<BooleanType>();
      result.addTypes(boolType);
    }]>
  ];
}

def eir_CmpEqOp :
    eir_BinaryComparisonOp<eir_AnyTerm, "cmp.eq", [Commutative]> {
  let summary = [{term equality comparison operation}];
  //let hasFolder = 1;

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value lhs, Value rhs, bool strict",
    [{
      result.addOperands(lhs);
      result.addOperands(rhs);
      auto boolType = builder->getType<BooleanType>();
      result.addTypes(boolType);
      auto strictAttr = builder->getBoolAttr(strict);
      result.addAttribute("strict", strictAttr);
    }]>
  ];
}

def eir_CmpNeqOp :
    eir_BinaryComparisonOp<eir_AnyTerm, "cmp.neq", [Commutative]> {
  let summary = [{term inequality comparison operation}];
  //let hasFolder = 1;

  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value lhs, Value rhs, bool strict",
    [{
      result.addOperands(lhs);
      result.addOperands(rhs);
      auto boolType = builder->getType<BooleanType>();
      result.addTypes(boolType);
      auto strictAttr = builder->getBoolAttr(strict);
      result.addAttribute("strict", strictAttr);
    }]>
  ];
}

def eir_CmpLtOp :
    eir_BinaryComparisonOp<eir_AnyTerm, "cmp.lt"> {
  let summary = [{term less-than comparison operation}];
  //let hasFolder = 1;
}

def eir_CmpLteOp :
    eir_BinaryComparisonOp<eir_AnyTerm, "cmp.lte"> {
  let summary = [{term less-than-or-equal comparison operation}];
  //let hasFolder = 1;
}

def eir_CmpGtOp :
    eir_BinaryComparisonOp<eir_AnyTerm, "cmp.gt"> {
  let summary = [{term greater-than comparison operation}];
  //let hasFolder = 1;
}

def eir_CmpGteOp :
    eir_BinaryComparisonOp<eir_AnyTerm, "cmp.gte"> {
  let summary = [{term greater-than-or-equal comparison operation}];
  //let hasFolder = 1;
}

def eir_CmpNerrOp :
    eir_UnaryComparisonOp<eir_AnyTerm, "cmp.nerr"> {
  let summary = [{term not-an-error comparison operation}];
  let description = [{
    Compares the given term against the NONE value, condition is true if not NONE
  }];
  //let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Control Flow
//===----------------------------------------------------------------------===//

def eir_BranchOp : eir_Op<"br", [Terminator]> {
  let summary = [{unconditional branch operation}];
  let description = [{
    Represents an unconditional branch operation that branches to a target block
    with the given set of arguments.

    ```
    ^bb0(...):
      eir.br ^bb1(%a)
    ^bb1(%blockArg1):
      ...
   ```
  }];

  let arguments = (ins
    Variadic<eir_AnyTerm>:$operands
  );

  let builders = [
    OpBuilder<[{
      Builder *, OperationState &result, Block *dest,
      ValueRange operands = {}
    }], [{
      result.addSuccessor(dest, operands);
    }]>,
   ];

  let extraClassDeclaration = [{
    Block *getDest();
    Block *dest() { return getDest(); }
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];
}

def eir_CondBranchOp : eir_Op<"cond_br", [Terminator]> {
  let summary = [{conditional branch operation}];
  let description = [{
    Represents a conditional branch operation that branches to one of the two
    target blocks with the given set of arguments.

    ```
    ^bb0(...):
      eir.cond_br %condition, ^bb1(%a), ^bb2(%b)
    ^bb1(%blockArg1):
      ...
    ^bb2(%blockArg2):
      ...
   ```
  }];

  let arguments = (ins
    eir_BoolType:$condition,
    Variadic<eir_AnyTerm>:$branchOperands
  );

  let builders = [
    OpBuilder<[{
      Builder *, OperationState &result, Value condition,
      Block *trueDest, ValueRange trueOperands,
      Block *falseDest, ValueRange falseOperands
    }], [{
      result.addOperands(condition);
      result.addSuccessor(trueDest, trueOperands);
      result.addSuccessor(falseDest, falseOperands);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    /// The condition operand is the first operand in the list.
    Value getCondition() { return getOperand(0); }

    /// Return the destination if the condition is true.
    Block *getTrueDest() {
      return getOperation()->getSuccessor(trueIndex);
    }

    /// Return the destination if the condition is false.
    Block *getFalseDest() {
      return getOperation()->getSuccessor(falseIndex);
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

    operand_iterator true_operand_begin() {
      return operand_begin() + getTrueDestOperandIndex();
    }
    operand_iterator true_operand_end() {
      return true_operand_begin() + getNumTrueOperands();
    }
    operand_range getTrueOperands() {
      return {true_operand_begin(), true_operand_end()};
    }

    unsigned getNumTrueOperands()  {
      return getOperation()->getNumSuccessorOperands(trueIndex);
    }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      getOperation()->eraseSuccessorOperand(trueIndex, index);
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

    operand_iterator false_operand_begin() { return true_operand_end(); }
    operand_iterator false_operand_end() {
      return false_operand_begin() + getNumFalseOperands();
    }
    operand_range getFalseOperands() {
      return {false_operand_begin(), false_operand_end()};
    }

    unsigned getNumFalseOperands() {
      return getOperation()->getNumSuccessorOperands(falseIndex);
    }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      getOperation()->eraseSuccessorOperand(falseIndex, index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 1; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];

  //let hasCanonicalizer = 1;
}

class eir_CallBaseOp<string mnemonic, list<OpTrait> traits = []> :
    eir_Op<mnemonic, !listconcat(traits, [CallOpInterface])> {
  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
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

def eir_CallOp : eir_CallBaseOp<"call"> {
  let summary = [{call operation}];
  let description = [{
    Calls a function with the given arguments.
  }];

  let arguments = (ins
    eir_FuncRefAttr:$callee,
    Variadic<eir_AnyTerm>:$operands
  );
  let results = (outs
    Variadic<eir_AnyTerm>:$results
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

def eir_ReturnOp : eir_Op<"return", [
    HasParent<"eir::FuncOp">,
    Terminator,
  ]> {
  let summary = "return operation";
  let description = [{
    Represents a return operation within a function.

    ```
    eir.func @foo(%0, %1) : (i32, f8) {
      eir.return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins
    Variadic<eir_AnyTerm>:$operands
  );

  let verifier = [{ return success(); }];

  let builders = [
    OpBuilder<[{
      Builder *b, OperationState &result
    }], [{
      build(b, result, llvm::None);
    }]>,
  ];
}

def eir_YieldOp : eir_Op<"yield", [
    HasParent<"eir::FuncOp">,
    YieldPoint,
  ]> {
  let summary = [{scheduler yield operation}];
  let description = [{
    Yields execution back to the scheduler
  }];

  let verifier = [{ return success(); }];
}

def eir_IfOp : eir_Op<"if", [Terminator]> {
  let summary = "if-then-else operation";
  let description = [{
    The "eir.if" operation represents an if-then-else construct for
    conditionally executing two regions of code, with an optional fallback region.
    The operand to an `if` operation is a term value, and the `if` branch is taken when
    the term is the atom 'true', the `else` branch is taken when the term is the atom 'false',
    and the `otherwise` branch is taken when the value is not an atom.

    If the `otherwise` branch is not given, then it is implicitly unreachable.
    The operation is a terminator, and produces no results.

    For example:

       eir.if %b  {
         ...
       } else {
         ...
       } otherwise {
         ...
       }

    Or without the optional `else`:

       eir.if %b  {
         ...
       } else {

       }
  }];
  let arguments = (ins eir_BoolLike:$condition);
  let regions = (region SizedRegion<1>:$ifRegion, SizedRegion<1>:$elseRegion, AnyRegion:$otherwiseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
      "Builder *builder, OperationState &result, "
      "Value cond, bool withOtherwiseRegion">
  ];

  let extraClassDeclaration = [{
    OpBuilder getIfBodyBuilder() {
      assert(!ifRegion().empty() && "Unexpected empty 'if' region.");
      Block &body = ifRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
    OpBuilder getElseBodyBuilder() {
      assert(!elseRegion().empty() && "Unexpected empty 'else' region.");
      Block &body = elseRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
    OpBuilder getOtherwiseBodyBuilder() {
      assert(!otherwiseRegion().empty() && "Unexpected empty 'otherwise' region.");
      Block &body = otherwiseRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
  }];
}

def eir_MatchOp : eir_Op<"match", [Terminator]> {
  let summary = "A general pattern matching operation";

  let description = [{
    A high-level way of expressing chained conditional branches based on specific patterns
  }];

  let arguments = (ins AnyType:$arg);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let hasCanonicalizer = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, Value selector,"
              "ArrayRef<MatchBranch> branches,"
              "ArrayRef<NamedAttribute> attributes = {}">
  ];

  let extraClassDeclaration = [{
    // The input value for the match
    Value getSelector() { return getOperation()->getOperand(0); }

    // The number of conditions that may be tested
    unsigned getNumConditions() { return getNumDest(); }

    // The number of blocks that may be branched to
    unsigned getNumDest() { return getOperation()->getNumSuccessors(); }
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

  let arguments = (ins);
  let results = (outs);

  let parser = [{ return mlir::success(); }];
  let printer = [{ p << getOperationName(); }];
  let verifier = [{ return mlir::success(); }];
}

def eir_ThrowOp : eir_Op<"throw", [Terminator]> {
  let summary = "Used to throw an exception at the current point of execution";
  let description = [{
    A corollary to `eir.return`, this function terminates execution of
    the current function, returning control up the stack by unwinding.
  }];
  let arguments = (ins eir_AnyTerm:$reason);

  let verifier = [{ return success(); }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

//===----------------------------------------------------------------------===//
// Miscellaneous Operations
//===----------------------------------------------------------------------===//

def CastOp : eir_Op<"cast", [NoSideEffect]> {

  let arguments = (ins eir_AnyTerm:$input);
  let results = (outs eir_AnyTerm:$output);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value source, Type destType", [{
       build(builder, result, source, destType);
  }]>];

  let assemblyFormat = [{
    $input `:` type($input) `to` type($output) attr-dict
  }];

  //let hasFolder = 1;
}

def eir_LoadOp : eir_Op<"load", []> {
  let summary = "Load a value from a memory reference";

  let description = [{
    Load a value from a memory reference into a virtual register.  Produces
    an immutable ssa-value of the referent type.
  }];

  let arguments = (ins eir_BoxType:$ref);
  let results = (outs eir_AnyTerm:$out);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value refVal",
    [{
      if (!refVal) {
        return;
      }
      if (auto boxed = refVal.getType().dyn_cast_or_null<eir::BoxType>()) {
        result.addOperands(refVal);
        result.addTypes(boxed.getBoxedType());
      }
    }]
  >];

  let verifier = [{
    auto refVal = ref();
    if (!refVal) {
        return emitError("invalid value: referenced value must not be null");
    }
    if (!refVal.getType().isa<eir::BoxType>()) {
        return emitError("invalid value: expected referenced type to be a boxed type");
    }
    return success();
  }];

  let assemblyFormat = [{
    `(` $ref `)` attr-dict `:` functional-type($ref, $out)
  }];
}

def eir_GetElementPtrOp : eir_Op<"getelementptr", [NoSideEffect]> {
  let summary = "An aggregate access primitive used to calculate memory addresses to an element of an aggregate type";
  let description = [{
    An aggregate access primitive used to calculate memory addresses to an element of an aggregate type
  }];

  let arguments = (ins eir_AnyTerm:$base, Variadic<eir_AnyTerm>:$indices);
  let results = (outs eir_AnyTerm:$res);

  let builders = [
    OpBuilder<
    "Builder *, OperationState &result, Type resultType, "
    "ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
    [{
      if (resultType) result.addTypes(resultType);
      result.addOperands(operands);
      for (auto namedAttr : attributes) {
        result.addAttribute(namedAttr.first, namedAttr.second);
      }
    }]>
  ];

  let verifier = [{ return success(); }];

  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` functional-type(operands, results)
  }];
}

def eir_PrintOp : eir_Op<"intrinsics.print"> {
  let summary = [{intrinsic printing operation}];
  let description = [{
    Prints the given string message and zero or more values.
  }];

  let arguments = (ins
    StrAttr:$message,
    Variadic<eir_AnyTerm>:$operands
  );

  let verifier = [{ return success(); }];

  let assemblyFormat = [{
    $message `(` operands `)` attr-dict `:` type(operands)
  }];
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

class eir_ConstantOp<string mnemonic> : eir_Op<mnemonic,
    [NoSideEffect]> {

  let skipDefaultBuilders = 1;
  let summary = "Constructs a constant term value";

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let parser = [{ return parseConstantOp(parser, result); }];
  let printer = [{ printConstantOp(p, *this); }];
  let verifier = [{ return verifyConstantOp(*this); }];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
  }];

  //let hasFolder = 1;
}

def eir_ConstantFloatOp : eir_ConstantOp<"constant.float"> {
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

def eir_ConstantIntOp : eir_ConstantOp<"constant.int"> {
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
      auto type = builder->getType<FixnumType>();
      auto attr = builder->getIntegerAttr(type, value);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantBigIntOp : eir_ConstantOp<"constant.bigint"> {
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

def eir_ConstantAtomOp : eir_ConstantOp<"constant.atom"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, APInt id, StringRef name",
    [{
      auto type = builder->getType<AtomType>();
      auto attr = AtomAttr::get(builder->getContext(), id, name);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantBinaryOp : eir_ConstantOp<"constant.binary"> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, Attribute val",
    [{
      result.addAttribute("value", val);
      result.addTypes(type);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result,"
    "ArrayRef<char> value, uint64_t header, uint64_t flags",
    [{
      auto type = builder->getType<eir::BinaryType>();
      auto attr = BinaryAttr::get(type, value, header, flags);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantNilOp : eir_ConstantOp<"constant.nil"> {
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
      auto type = builder->getType<NilType>();
      auto attr = TypeAttr::get(type);
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantNoneOp : eir_ConstantOp<"constant.none"> {
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
      auto type = builder->getType<mlir::NoneType>();
      auto attr = builder->getUnitAttr();
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantTupleOp : eir_ConstantOp<"constant.tuple"> {
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

def eir_ConstantListOp : eir_ConstantOp<"constant.list"> {
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
      build(builder, result, type, attr);
    }]>
  ];
}

def eir_ConstantMapOp : eir_ConstantOp<"constant.map"> {
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

  let arguments = (ins eir_AnyTerm:$head, eir_AnyTerm:$tail);
  let results = (outs eir_ConsType:$out);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, Value head, Value tail">
  ];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, $out)
  }];
}

def eir_TupleOp : eir_Op<"tuple"> {
  let summary = "Tuple constructor";
  let description = [{
    Tuple construction primitive. Constructs a new tuple term from a list of elements.
  }];

  let arguments = (ins Variadic<eir_AnyTerm>:$elements);
  let results = (outs eir_TupleType:$out);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, ArrayRef<Value> values">
  ];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, $out)
  }];
}

def eir_TraceCaptureOp : eir_Op<"trace_capture", [Terminator]> {
  let summary = "Captures the current stack trace";
  let description = [{
    This operation is called when raising an error to inform the process
    to capture and store the current stack trace until a landing pad is
    reached.

    Meant to be used in conjunction with `trace_construct` to access the
    captured trace as a term.

        func @test(!eir.term) -> !eir.boolean {
        ^entry(%val : !eir.term):
            (%0, %err) = eir.call @try_test(%val) : (!eir.boolean, i1)
            cond_br %err, ^continue(%0: !eir.boolean), ^failed
        ^continue(%passed):
            return %passed: !eir.boolean
        ^failed:
            eir.trace_capture
            eir.throw
        }
  }];

  let builders = [
    OpBuilder<[{
      Builder *, OperationState &result, Block *dest, ValueRange destArgs = {}
      }], [{
      result.addSuccessor(dest, destArgs);
    }]>,
  ];
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

  let arguments = (ins eir_AnyTerm:$capture);
  let results = (outs eir_AnyTerm:$trace);

  let assemblyFormat = [{
    `(` $capture `)` attr-dict `:` functional-type($capture, $trace)
  }];
}

def eir_ConstructMapOp : eir_Op<"map.new"> {
  let summary = "Map constructor";
  let description = [{
    Map construction primitive. Creates a new map from a list of key/value pairs.
  }];

  let arguments = (ins Variadic<eir_AnyTerm>:$args);
  let results = (outs eir_MapType:$out, eir_AnyTerm:$err);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, ArrayRef<eir::MapEntry> entries">
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

  let arguments = (ins eir_MapType:$map, Variadic<eir_AnyTerm>:$args);
  let results = (outs eir_MapType:$newMap, eir_AnyTerm:$err);

  let verifier = [{ return success(); }];

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

  let arguments = (ins eir_MapType:$map, Variadic<eir_AnyTerm>:$args);
  let results = (outs eir_MapType:$newMap, eir_AnyTerm:$err);

  let verifier = [{ return success(); }];

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

def eir_BinaryPushOp : eir_Op<"binary.push"> {
  let summary = "Pushes a value into a binary based on the given specifier";
  let description = [{
    Used to construct a binary piece by piece.

    Each invocation appends a value to the binary based on a binary specification,
    provided as attributes to the operation.

        %0, %err = eir.binary_push(%bin, %val) { type = integer, signed = true, endian = big, unit = 1 }
        %0, %err = eir.binary_push(%bin, %val) { type = bytes, unit = 1 }, %size : !eir.fixnum
  }];

  let arguments = (ins eir_BinaryType:$bin, eir_AnyTerm:$val, Variadic<eir_AnyTerm>:$size);
  let results = (outs eir_BinaryType:$updatedBin);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, $updatedBin)
  }];
}

#endif // EIR_OPS
