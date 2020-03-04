#ifndef EIR_OPS
#define EIR_OPS

include "mlir/Analysis/CallInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "lumen/compiler/Dialect/EIR/IR/EIRBase.td"

def eir_FuncOp : eir_Op<"func", [
    IsolatedFromAbove,
    //HasParent<"ModuleOp">,
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
      auto boolType = builder->getType<::lumen::eir::BooleanType>();
      result.addTypes(boolType);
    }]>
  ];
}

def eir_CmpEqOp :
    eir_BinaryComparisonOp<eir_AnyType, "cmp.eq", [Commutative]> {
  let summary = [{term equality comparison operation}];
  //let hasFolder = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value lhs, Value rhs, bool strict",
    [{
      result.addOperands(ValueRange({lhs, rhs}));
      auto boolType = builder->getType<::lumen::eir::BooleanType>();
      result.addTypes(boolType);
      auto strictAttr = builder->getBoolAttr(strict);
      result.addAttribute("strict", strictAttr);
    }]>
  ];
}

def eir_CmpNeqOp :
    eir_BinaryComparisonOp<eir_AnyType, "cmp.neq", [Commutative]> {
  let summary = [{term inequality comparison operation}];
  //let hasFolder = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value lhs, Value rhs, bool strict",
    [{
      result.addOperands(lhs);
      result.addOperands(rhs);
      auto boolType = builder->getType<::lumen::eir::BooleanType>();
      result.addTypes(boolType);
      auto strictAttr = builder->getBoolAttr(strict);
      result.addAttribute("strict", strictAttr);
    }]>
  ];
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
    Variadic<eir_AnyType>:$operands
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
    Variadic<eir_AnyType>:$branchOperands
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

def eir_ReturnOp : eir_Op<"return", [
    //HasParent<"mlir::FuncOp">,
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
    Variadic<eir_AnyType>:$operands
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
    HasParent<"mlir::FuncOp">,
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
  let arguments = (ins eir_AnyType:$reason);

  let verifier = [{ return success(); }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

//===----------------------------------------------------------------------===//
// Miscellaneous Operations
//===----------------------------------------------------------------------===//

def eir_CastOp : eir_Op<"cast", [NoSideEffect]> {

  let arguments = (ins eir_AnyType:$input);
  let results = (outs eir_AnyType:$output);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, Value source, Type destType",
    [{
      result.addOperands(source);
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

  let arguments = (ins Variadic<eir_AnyType>:$index,
                   Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment);
  let results = (outs eir_BoxType);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, BoxType boxType, "
              "IntegerAttr alignment = IntegerAttr()",
    [{
       result.types.push_back(boxType);
       if (alignment)
         result.addAttribute(getAlignmentAttrName(), alignment);
    }]>,
    OpBuilder<"Builder *builder, OperationState &result, BoxType boxType, "
              "ArrayRef<Value> operands, IntegerAttr alignment = IntegerAttr()",
    [{
       result.addOperands(operands);
       result.types.push_back(boxType);
       if (alignment)
         result.addAttribute(getAlignmentAttrName(), alignment);
    }]>
  ];

  let extraClassDeclaration = [{
    static StringRef getAlignmentAttrName() { return "alignment"; }

    BoxType getType() { return getResult().getType().cast<BoxType>(); }

    /// Returns the dynamic sizes for this alloc operation if specified.
    int64_t getAllocSize(unsigned pointerSizeInBits) {
      return calculateAllocSize(pointerSizeInBits, getType());
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

def eir_GetElementPtrOp : eir_Op<"getelementptr", [NoSideEffect]> {
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
        return;
      }
      auto termTy = builder->getType<eir::TermType>().cast<OpaqueTermType>();
      if (elementTy.isa<ConsType>()) {
        auto refTy = builder->getType<eir::RefType>(termTy);
        result.addTypes(refTy);
      } else if (auto tupleTy = elementTy.dyn_cast_or_null<TupleType>()) {
        if (tupleTy.hasDynamicShape()) {
          auto refTy = builder->getType<eir::RefType>(termTy);
          result.addTypes(refTy);
        } else if (tupleTy.getArity() > index) {
          auto innerTy = tupleTy.getElementType(index).cast<OpaqueTermType>();
          auto refTy = builder->getType<eir::RefType>(innerTy);
          result.addTypes(refTy);
        } else {
          auto refTy = builder->getType<eir::RefType>(termTy);
          result.addTypes(refTy);
        }
      } else {
        auto refTy = builder->getType<eir::RefType>(termTy);
        result.addTypes(refTy);
      }
      result.addOperands(boxedVal);
      auto indexTy = builder->getIndexType();
      result.addAttribute("index", builder->getIntegerAttr(indexTy, index));
    }]>
  ];

  let verifier = [{ return success(); }];

  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    static StringRef getIndexAttrName() { return "index"; }

    Type getType() { return getResult().getType().cast<RefType>().getInnerType(); }

    uint64_t getIndex() {
        return getAttrOfType<IntegerAttr>("index").getValue().getLimitedValue();
    }
  }];

}

def eir_PrintOp : eir_Op<"intrinsics.print"> {
  let summary = "intrinsic printing operation";
  let description = [{
    Prints to standard output using the provided format string and operands.
  }];

  let arguments = (ins
    Variadic<eir_AnyType>:$operands
  );
  let results = (outs eir_AnyTerm:$result);

  let verifier = [{ return success(); }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

class eir_ConstantOp<Type type, string mnemonic> : eir_Op<mnemonic,
    [NoSideEffect]> {

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

  //let hasFolder = 1;
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
      auto attr = builder->getUnitAttr();
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

def eir_ConstantListOp : eir_ConstantOp<eir_ConsType, "constant.list"> {
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
  let results = (outs eir_ConsType:$out);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, Value head, Value tail">
  ];

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
  let results = (outs eir_TupleType:$out);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, ArrayRef<Value> values">
  ];

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

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($capture)
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

  let arguments = (ins);
  let results = (outs eir_AnyTerm:$trace);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($trace)
  }];
}

def eir_ConstructMapOp : eir_Op<"map.new"> {
  let summary = "Map constructor";
  let description = [{
    Map construction primitive. Creates a new map from a list of key/value pairs.
  }];

  let arguments = (ins Variadic<eir_AnyType>:$args);
  let results = (outs eir_MapType:$out, eir_AnyType:$err);

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

  let arguments = (ins eir_MapType:$map, Variadic<eir_AnyType>:$args);
  let results = (outs eir_MapType:$newMap, eir_AnyType:$err);

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

  let arguments = (ins eir_MapType:$map, Variadic<eir_AnyType>:$args);
  let results = (outs eir_MapType:$newMap, eir_AnyType:$err);

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

  let arguments = (ins eir_BinaryType:$bin, eir_AnyType:$val, Variadic<eir_AnyType>:$size);
  let results = (outs eir_BinaryType:$updatedBin);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, $updatedBin)
  }];
}

#endif // EIR_OPS
