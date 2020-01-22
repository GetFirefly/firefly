//===- Ops.td - EIR operation definitions ----------*- tablegen -*-===//
//
// Defines the operations of the Toy dialect.
//
//===--------------------------------------------------------------===//
#ifndef EIR_OPS
#define EIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Analysis/CallInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "Base.td"
include "Types.td"

//===----------------------------------------------------------------------===//
// EIR Operations
//
// Every operation is defined by inheriting from a base (i.e. `eir_Op`) operation
// class; and defines the mnemonic, traits, inputs, outputs, builder, verifier,
// and other important details of the operation being defined.
//
// Traits:
//
// - NoSideEffect: a pure operation that has no side effects (including writing memory)
// - Commutative: X + Y == Y + X
// - ResultsAreFloatLike: result of an op is a float, or a vector/tensor of floats
// - Symbol: operation is a symbol
// - SymbolTable: operation defines a symbol table
// - Terminator: operation is a terminator, i.e. last operation in a block
// - HasParent: operation is required to have the given parent op
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

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
  let arguments = (ins AnyBooleanLike:$condition);
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

def eir_MatchOp : eir_Op<"match", [InFunctionScope, Terminator]> {
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

def eir_CallOp : eir_Op<"call", [
    InFunctionScope, Terminator, DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Call a function";

  let description = [{
    Provides a custom parser and pretty printer to allow a more readable syntax
    in the EIR dialect, e.g. `eir.call @sub(%12)` or `eir.call %20(%22,%23)`.

    ## Assembly

    ``` {.ebnf}
    function-call-op ::= `eir.call` function-id `(` ssa-use-list `)` `:` function-type
    ```

    ## Usage

    ```
    eir.call @foo(%arg0) : (i32) -> ()
    %0 = eir.call @double(%arg0, %arg1) : (i32, i32) -> i32
    ```
  }];

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result,"
    "SymbolRefAttr callee, ArrayRef<Value> args, bool isTail,"
    "Block *ok, ArrayRef<Value> okArgs, Block *err, ArrayRef<Value> errArgs",
    [{
      result.addOperands(args);

      if (callee) {
        result.addAttribute("callee", callee);
      }
      if (ok) {
        result.addSuccessor(ok, okArgs);
      } else {
        result.addAttribute("return", builder->getBoolAttr(true));
      }
      if (err) {
        result.addSuccessor(err, errArgs);
      } else {
        result.addAttribute("throw", builder->getBoolAttr(true));
      }
      result.addAttribute("is_tail", builder->getBoolAttr(isTail));
    }]
  >];

  let arguments = (ins
    OptionalAttr<SymbolRefAttr>:$callee,
    Variadic<AnyType>:$arguments
  );

  let results = (outs);

  let extraClassDeclaration = [{
    M::FunctionType getCalleeType();

    bool returnsOnSuccess() {
      if (auto val = getAttrOfType<mlir::BoolAttr>("return"))
        return val.getValue();
      return false;
    }

    bool throwsOnError() {
      if (auto val = getAttrOfType<mlir::BoolAttr>("throw"))
        return val.getValue();
      return false;
    }

    bool isTailCall() {
      if (auto val = getAttrOfType<mlir::BoolAttr>("is_tail"))
        return val.getValue();
      return false;
    }
  }];
}

def eir_UnreachableOp : eir_Op<"unreachable", [InFunctionScope, Terminator]> {
  let summary = "Used to inform the compiler that this point in the CFG is unreachable";

  let description = [{
    Terminates a basic block with the assertion that the end of the block
    will never be reached at runtime.  This instruction can be used
    immediately after a call to the runtime to terminate the program.
  }];

  let arguments = (ins);
  let results = (outs);

  let parser = [{ return M::success(); }];
  let printer = [{ p << getOperationName(); }];
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

class eir_AllocatableOp<string mnemonic, list<OpTrait> traits =[]> :
    eir_AllocatableBaseOp<mnemonic, !listconcat([NoSideEffect], traits)>,
    eir_TwoBuilders<eir_AllocateOpBuilder, eir_NamedAllocateOpBuilder>,
    Arguments<(ins TypeAttr:$in_type, Variadic<AnyIntegerType>:$args)> {

  let parser = [{ return parseAllocatableOp<$cppClass>(parser, result); }];
  let printer = [{ return printAllocatableOp(p, *this); }];

  string extraAllocClassDeclaration = [{
    constexpr static llvm::StringRef lenpName() { return "len_param_count"; }

    mlir::Type getAllocatedType();
    bool hasLenParams() { return bool{getAttr(lenpName())}; }
    unsigned numLenParams() {
      if (auto val = getAttrOfType<mlir::IntegerAttr>(lenpName()))
        return val.getInt();
      return 0;
    }
    operand_range getLenParams() {
      return {operand_begin(), operand_begin() + numLenParams()};
    }
    operand_range getShapeOperands() {
      return {operand_begin() + numLenParams(), operand_end()};
    }
    static mlir::Type getRefTy(mlir::Type ty);
  }];
}

def eir_AllocaOp : eir_AllocatableOp<"alloca"> {
  let summary = "Allocate storage for a temporary on the stack given a type";

  let description = [{
    This primitive operation is used to allocate an object on the stack.  A
    reference to the object of type `!eir.ref<T>` is returned.  The returned
    object has an undefined state.  The allocation can be given an optional
    name.  The allocation may have a dynamic repetition count for allocating
    a sequence of locations for the specified type.

      %0 = eir.alloca !eir.float
      %1 = eir.call @foo(%0) : (!eir.ref<!eir.float>) -> !eir.float
  }];

  let verifier = [{
    M::Type outType = getType();
    if (!outType.dyn_cast<eir::RefType>())
      return emitOpError("must be a !eir.ref type");
    return M::success();
  }];

  let extraClassDeclaration = extraAllocClassDeclaration#[{
    static mlir::Type wrapResultType(mlir::Type intype);
  }];
}

def eir_LoadOp : eir_OneResultOp<"load", []>,
    Arguments<(ins AnyRefLike:$memref)> {
  let summary = "Load a value from a memory reference";

  let description = [{
    Load a value from a memory reference into a virtual register.  Produces
    an immutable ssa-value of the referent type.
  }];

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value refVal",
    [{
      if (!refVal) {
        mlir::emitError(result.location, "LoadOp has null argument");
        return;
      }
      eir::RefType refTy = refVal.getType().cast<eir::RefType>();
      result.addOperands(refVal);
      result.addTypes(refTy.getEleTy());
    }]
  >];

  let extraClassDeclaration = [{
    static mlir::ParseResult getElementOf(mlir::Type &ele, mlir::Type ref);
  }];
}

def eir_StoreOp : eir_Op<"store", []>,
    Arguments<(ins AnyType:$value, AnyRefLike:$memref)> {
  let summary = "Store an SSA-value to a memory location";

  let description = [{
    Store an ssa-value (virtual register) to a memory reference.

    The stored value must be of the same type as the referent type of the memory reference.
  }];

  let extraClassDeclaration = [{
    static mlir::Type elementType(mlir::Type refType);
  }];
}

def eir_UndefOp : eir_OneResultOp<"undefined", [NoSideEffect]> {
  let summary = "Explicit undefined value of some type";

  let description = [{
    Constructs an ssa-value of the specified type with an undefined value.
    This operation is typically created internally by the mem2reg conversion
    pass.
  }];
}

def eir_MallocOp : eir_AllocatableOp<"malloc"> {
  let summary = "Allocate storage on the process heap for an object of a given type";

  let description = [{
    Creates a heap memory reference suitable for storing a value of the given type, T.

    The heap reference returned has type `!eir.box<T>`.

    The object is in an undefined state, until stored to.

      %0 = eir.malloc !eir.tuple<10x!eir.float>
  }];

  let extraClassDeclaration = extraAllocClassDeclaration#[{
    static mlir::Type wrapResultType(mlir::Type intype);
  }];
}

//===----------------------------------------------------------------------===//
// Miscellaneous Operations
//===----------------------------------------------------------------------===//

def eir_ConstantOp : eir_Op<"constant",
    [NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface>]> {

  let summary = "Constructs a constant term value";

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Attribute value",
    [{ build(builder, result, value.getType(), value); }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }

    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Runtime Operations
//===----------------------------------------------------------------------===//

def eir_ConsOp : eir_Op<"cons", [InFunctionScope]> {
  let summary = "Cons operation for constructing list cells.";
  let description = [{
    List construction primitive. Constructs a new list cell from head and tail terms.
  }];

  let arguments = (ins AnyType:$head, AnyType:$tail);
  let results = (outs eir_ConsType:$out);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, Value head, Value tail">
  ];
}

def eir_TupleOp : eir_Op<"tuple", [InFunctionScope]> {
  let summary = "Tuple constructor";
  let description = [{
    Tuple construction primitive. Constructs a new tuple term from a list of elements.
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs eir_TupleType:$out);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, ArrayRef<Value> values">
  ];
}

def eir_CastOp : eir_Op<"cast", [NoSideEffect, InFunctionScope]> {
  let summary = "A bitcasting primitive used to cast to/from one term type to another.";
  let description = [{
    A bitcasting primitive used to cast to/from one term type to another.
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$out);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, Value value, Type outType",
    [{
      result.addOperands(value);
      result.types.push_back(outType);
    }]>,
  ];

  let extraClassDeclaration = [{
    static bool areCastCompatible(M::Type inType, M::Type outType);

    M::Type getInputType() { return getOperand().getType(); }
    M::Type getResultType() { return getResult().getType(); }
  }];
}

def eir_GetElementOp : eir_Op<"get_element", [NoSideEffect, InFunctionScope]> {
  let summary = "An aggregate access primitive used to calculate memory addresses to an element of an aggregate type";
  let description = [{
    An aggregate access primitive used to calculate memory addresses to an element of an aggregate type
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$out);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, Value value, Type outType, M::IntegerAttr index",
    [{
      result.addOperands(value);
      result.types.push_back(outType);
      result.addAttribute("index", index);
    }]>,
    OpBuilder<"Builder *builder, OperationState &result, Value value, Type outType, unsigned index",
    [{
      M::Type indexType = builder->getIndexType();
      M::IntegerAttr indexAttr = builder->getIntegerAttr(indexType, (int64_t)index);
      build(builder, result, value, outType, indexAttr);
    }]>,
    OpBuilder<"Builder *builder, OperationState &result, Value value, unsigned index",
    [{ build(builder, result, value, builder->getType<TermType>(), index); }]>
  ];

  let extraClassDeclaration = [{
    Value getAggregate() { return getOperation()->getOperand(0); }
    Type getAggregateType() { return getOperation()->getOperand(0).getType(); }
    Type getResultType() { return getResult().getType(); }
    M::IntegerAttr getIndexAttr() { return getAttrOfType<M::IntegerAttr>("index"); }
    unsigned getIndex() { return getIndexAttr().getValue().getLimitedValue(); }
  }];
}

def eir_IsTypeOp : eir_Op<"is_type", [NoSideEffect, InFunctionScope]> {
  let summary = "A type checking primitive used in conditional branching.";

  let description = [{
    This operation is used to examine the type of a term value and return a
    boolean or boolean term indicating whether the provided type matches that
    of the value.
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyBooleanLike:$isMatch);

  let hasCanonicalizer = 1;

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
    [{ build(builder, result, value, matchType, builder->getI1Type()); }]>
  ];

  let extraClassDeclaration = [{
    M::Type getValueType() {
      return getOperand().getType();
    }

    M::Type getMatchType() {
      Attribute attr = getAttr("type");
      TypeAttr typeAttr = attr.dyn_cast<TypeAttr>();
      return typeAttr.getValue();
    }

    M::Type getResultType() { return getResult().getType(); }
  }];
}

class eir_LogicalOp<string mnemonic, list<OpTrait> traits = []> : eir_Op<mnemonic, traits> {
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, "
    "Value lhs, Value rhs, Type returnType",
    [{
        result.addOperands(lhs);
        result.addOperands(rhs);
        result.addTypes(returnType);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, "
    "Value lhs, Value rhs",
    [{ build(builder, result, lhs, rhs, builder->getI1Type()); }]>
  ];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyBooleanLike:$out);

  let parser = [{ return parseLogicalOp(parser, result); }];
  let printer = [{ return printLogicalOp(p, *this); }];
  let verifier = [{ return verifyLogicalOp(*this); }];
}

class eir_ComparisonOp<string mnemonic, list<OpTrait> traits = []> : eir_Op<mnemonic, traits> {
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, "
    "Value lhs, Value rhs, Type returnType",
    [{
        result.addOperands(lhs);
        result.addOperands(rhs);
        result.addTypes(returnType);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, "
    "Value lhs, Value rhs",
    [{ build(builder, result, lhs, rhs, builder->getI1Type()); }]>
  ];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyBooleanLike:$out);

  let parser = [{ return parseComparisonOp(parser, result); }];
  let printer = [{ return printComparisonOp(p, *this); }];
  let verifier = [{ return verifyComparisonOp(*this); }];
}

class eir_EqualityOp<string mnemonic, list<OpTrait> traits = []> : eir_ComparisonOp<mnemonic, traits> {
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, "
    "Value lhs, Value rhs, Type returnType, bool isExact",
    [{
        result.addOperands(lhs);
        result.addOperands(rhs);
        result.addTypes(returnType);
        result.addAttribute("exact", builder->getBoolAttr(isExact));
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, "
    "Value lhs, Value rhs, bool isExact",
    [{ build(builder, result, lhs, rhs, builder->getI1Type(), isExact); }]>
  ];
}

def eir_IsEqualOp : eir_EqualityOp<"is_equal", [NoSideEffect, InFunctionScope]> {
  let summary = "equality comparison";
}

def eir_IsNotEqualOp : eir_EqualityOp<"is_not_equal", [NoSideEffect, InFunctionScope]> {
  let summary = "inequality comparison";
}

def eir_IsLessThanOrEqualOp : eir_ComparisonOp<"is_lte", [NoSideEffect, InFunctionScope]> {
  let summary = "=< comparison";
}

def eir_IsLessThanOp : eir_ComparisonOp<"is_lt", [NoSideEffect, InFunctionScope]> {
  let summary = "< comparison";
}

def eir_IsGreaterThanOrEqualOp : eir_ComparisonOp<"is_gte", [NoSideEffect, InFunctionScope]> {
  let summary = ">= comparison";
}

def eir_IsGreaterThanOp : eir_ComparisonOp<"is_gt", [NoSideEffect, InFunctionScope]> {
  let summary = "> comparison";
}

def eir_LogicalAndOp : eir_LogicalOp<"and", [NoSideEffect, InFunctionScope]> {
  let summary = "logical AND";
}

def eir_LogicalOrOp : eir_LogicalOp<"or", [NoSideEffect, InFunctionScope]> {
  let summary = "logical OR";
}

def eir_TraceCaptureOp : eir_Op<"trace_capture", []> {
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
}

def eir_TraceConstructOp : eir_Op<"trace_construct", []>, Results<(outs eir_TermType)> {
  let summary = "Constructs the current stack trace as a new SSA-value";
  let description = [{
    This operation should be called prior to generating an error which
    needs the current stack trace. The value returned will be an opaque
    term from the perspective of MLIR, but the runtime can use it like
    a pointer to the raw trace.

        %0 = eir.trace_construct
  }];
}

class eir_MapOp<string mnemonic, list<OpTrait> traits = []> :
    eir_Op<mnemonic, traits>, Results<(outs eir_MapType:$res, I1:$err)> {
  let builders = [eir_MapOpBuilder];
}

def eir_ConstructMapOp : eir_MapOp<"map", [InFunctionScope]> {
  let summary = "Map constructor";
  let description = [{
    Map construction primitive. Creates a new map from a list of key/value pairs.
  }];

  let arguments = (ins Variadic<AnyType>:$args);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, ArrayRef<eir::MapEntry> entries">
  ];
}

class eir_MapPutOp<string mnemonic, list<OpTrait> traits = []> : eir_MapOp<mnemonic, traits> {
  let builders = [eir_MapOpBuilder];
  let parser = [{ return parseMapPutOp(parser, result); }];
  let printer = [{ return printMapPutOp(p, *this); }];
  let verifier = [{ return verifyMapPutOp(*this); }];

  let arguments = (ins eir_MapType:$map, Variadic<eir_Type>:$args);
}

def eir_MapInsertOp : eir_MapPutOp<"map_insert", []> {
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
}

def eir_MapUpdateOp : eir_MapPutOp<"map_update", []> {
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
}

def eir_BinaryPushOp : eir_RuntimeOp<"binary_push", []> {
  let summary = "Pushes a value into a binary based on the given specifier";
  let description = [{
    Used to construct a binary piece by piece.

    Each invocation appends a value to the binary based on a binary specification,
    provided as attributes to the operation.

        %0, %err = eir.binary_push %bin, %val { type = integer, signed = true, endian = big, unit = 1 }
        %0, %err = eir.binary_push %bin, %val { type = bytes, unit = 1 }, %size : !eir.fixnum
  }];

  let arguments = (ins eir_BinaryType:$bin, eir_Type:$val);
  let results = (outs eir_BinaryType:$updatedBin, I1:$errorFlag);
}

#endif // EIR_OPS
