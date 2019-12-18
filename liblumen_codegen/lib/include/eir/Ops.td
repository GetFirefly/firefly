//===- Ops.td - EIR operation definitions ----------*- tablegen -*-===//
//
// Defines the operations of the Toy dialect.
//
//===--------------------------------------------------------------===//
#ifndef EIR_OPS
#define EIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Analysis/CallInterfaces.td"
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
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. The operation produces no results. For example:

       eir.if %b  {
         ...
       } else {
         ...
       }

    The 'else' block is optional, and may be omitted. For
    example:

       eir.if %b  {
         ...
       }
  }];
  let arguments = (ins AnyBooleanLike:$condition);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "Value *cond, bool withElseRegion">
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder() {
      assert(!thenRegion().empty() && "Unexpected empty 'then' region.");
      Block &body = thenRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
    OpBuilder getElseBodyBuilder() {
      assert(!elseRegion().empty() && "Unexpected empty 'else' region.");
      Block &body = elseRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
  }];
}

def eir_MergeOp : eir_Op<"merge", [HasParent<"MatchOp">, Terminator]> {
  let summary = "A special terminator for merging a structured selection.";

  let description = [{
    This operation is used to terminate a match block and jump to the merge
    point, which is the next operation following the match operation.

    This operation is automatically generated as part of lowering from EIR.
  }];

  let arguments = (ins);
  let results = (outs);
}

def eir_MatchOp : eir_Op<"match", [InFunctionScope]> {
  let summary = "A general pattern matching operation";

  let description = [{
    A high-level way of expressing chained conditional branches based on specific patterns
  }];

  let arguments = (ins);
  let results = (outs);

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    // Returns the match header block.
    Block *getHeaderBlock();

    // Returns the match merge block.
    Block *getMergeBlock();

    // Adds a merge block containing one eir.merge op.
    void addMergeBlock();
  }];

  let hasCanonicalizer = 1;
}

def eir_CallOp : eir_Op<"call", [
    InFunctionScope, DeclareOpInterfaceMethods<CallOpInterface>]> {
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

  let arguments = (ins
    OptionalAttr<SymbolRefAttr>:$callee,
    Variadic<AnyType>:$arguments
  );

  let results = (outs
    Variadic<AnyType>:$result
  );

  let extraClassDeclaration = [{
    M::FunctionType getCalleeType();
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
    "Builder *builder, OperationState &result, Value *refVal",
    [{
      if (!refVal) {
        mlir::emitError(result.location, "LoadOp has null argument");
        return;
      }
      eir::RefType refTy = refVal->getType().cast<eir::RefType>();
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
// Runtime Operations
//===----------------------------------------------------------------------===//

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

class eir_MapPutOp<string mnemonic, list<OpTrait> traits = []> : eir_Op<mnemonic, traits> {
  let builders = [eir_RuntimeOpBuilder];
  let parser = [{ return parseMapPutOp(parser, result); }];
  let printer = [{ return printMapPutOp(p, *this); }];
  let verifier = [{ return verifyMapPutOp(*this); }];

  let arguments = (ins eir_MapType:$map, Variadic<eir_Type>:$args);
  let results = (outs eir_MapType:$updatedMap, I1:$errorFlag);
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

  let arguments = (ins BinaryType:$bin, eir_Type:$val);
  let results = (outs BinaryType:$updatedBin, I1:$errorFlag);
}

#endif // EIR_OPS
