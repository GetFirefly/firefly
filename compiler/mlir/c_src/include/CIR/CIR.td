#ifndef CIR_OPS
#define CIR_OPS

include "CIR/Dialect.td"
include "CIR/Types.td"
include "CIR/Attributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CIR_CirEndOp : CIR_Op<"end", [Terminator]> {
  let summary = "A terminator instruction used in CIR ops";
  let description = [{
    An operation used with certain region-holding CIR ops that require a terminator where
    another terminator is not applicable, e.g. return.
  }];
}

def ImplicitCIRTerminator : SingleBlockImplicitTerminator<"CirEndOp">;

def CIR_DispatchTableOp : CIR_Op<"dispatch_table",
    [IsolatedFromAbove, ImplicitCIRTerminator]> {
  let summary = "Defines a dispatch table within the current module";
  let description = [{
    Defines a dispatch table associated with the given module name, that is joined together
    at link-time with all other dispatch tables to form the global dispatch table used by apply/3.

    The table itself is associated with a specific module, and entries in the table associate
    a function name and arity with a specific function symbol that will be called when dispatching
    to the combined MFA.
  }];

  let arguments = (ins FlatSymbolRefAttr:$module);
  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    $module attr-dict $region
  }];

  let hasRegionVerifier = 1;

  let extraClassDeclaration = [{
    void appendTableEntry(Operation *op);

    StringRef getModule() { return (*this)->getAttrOfType<FlatSymbolRefAttr>("module").getValue(); }
    Block &getBody() { return getRegion().front(); }
  }];
}

def CIR_DispatchEntryOp : CIR_Op<"dispatch_entry", [HasParent<"DispatchTableOp">]> {
  let summary = "Defines an entry in a dispatch table";
  let description = [{
    This op represents an entry in a dispatch table. It binds a function name and arity
    to a function symbol which should be called when applying an MFA for which the entry
    is a match.

    Each entry in a table must be a unique function/arity or verification will fail.

    ```mlir
    cir.dispatch_entry run, 1, @"module:run/1"
    ```
  }];

  let arguments = (ins StrAttr:$function, I8Attr:$arity, FlatSymbolRefAttr:$symbol);

  let assemblyFormat = [{
    $function `,` $arity `,` $symbol attr-dict
  }];

  let extraClassDeclaration = [{
    unsigned getArity() { return (*this)->getAttrOfType<IntegerAttr>("arity").getInt(); }
    StringRef getFunction() { return (*this)->getAttrOfType<StringAttr>("function").getValue(); }
    FlatSymbolRefAttr getSymbol() { return (*this)->getAttrOfType<FlatSymbolRefAttr>("symbol"); }
  }];
}

def CIR_CallOp : CIR_Op<"call",
    [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `cir.call` operation represents a direct call to a function that is
    within the same symbol scope as the call. The operands and result types of
    the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = cir.call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def CIR_CastOp : CIR_Op<"cast",
  [DeclareOpInterfaceMethods<CastOpInterface>, NoSideEffect]> {
  let summary = "Performs a type cast between the type of the input value and given output type";

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    ($inputs^ `:` type($inputs))? `to` type($outputs) attr-dict
  }];
  let hasFolder = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$outputs, "ValueRange":$inputs), [{
      SmallVector<Type, 1> inputTypes;
      for (auto input : inputs)
        inputTypes.push_back(input.getType());
      $_state.addOperands(inputs);
      $_state.addAttribute("src", $_builder.getTypeArrayAttr(inputTypes));
      $_state.addTypes(outputs);
    }]>,
  ];

  let extraClassDeclaration = [{
    static StringRef getSrcAttrName() { return "src"; }

    auto getInputTypes() const {
      return (*this)->getAttrOfType<ArrayAttr>(getSrcAttrName()).getAsValueRange<TypeAttr>();
    }
  }];
}

def CIR_ConstantOp : CIR_Op<"constant",
  [ConstantLike, NoSideEffect,
   TypesMatchWith<"result type matches value type", "value", "result", "$_self">]> {
    let summary = "constant or immediate term";
    let description = [{
      The `constant` operation produces an SSA value equal to some term constant specified by an attribute.
    }];

    let arguments = (ins AnyAttr:$value);
    let results = (outs CIR_AnyTerm:$result);

    let builders = [
      OpBuilder<(ins "Attribute":$value),
      [{ build($_builder, $_state, value.getType(), value); }]>,
      OpBuilder<(ins "Attribute":$value, "Type":$type),
      [{ build($_builder, $_state, type, value); }]>,
    ];

    let extraClassDeclaration = [{
      // Returns true if a constant can be constructed with the given value and type
      static bool isBuildableWith(Attribute value, Type type);
    }];

    let hasFolder = 1;
    let assemblyFormat = "$value attr-dict";
}

def CIR_ConstantNullOp : CIR_Op<"null", [NoSideEffect]> {
    let summary = "constant or immediate null";
    let description = [{
      The `constant` operation produces an SSA value equal to the null value of the given type
    }];

    let arguments = (ins);
    let results = (outs AnyType:$result);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<(ins "Type":$type), [{
        $_state.addTypes({type});
      }]>
    ];

    let extraClassDeclaration = [{
      // Returns true if a constant can be constructed with the given value and type
      static bool isBuildableWith(Attribute value, Type type);
    }];

    let assemblyFormat = "`:` type($result) attr-dict";
}

def CIR_AndOp : CIR_Op<"and", [NoSideEffect]> {
  let summary = "Performs a boolean AND operation";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
  }];
}

def CIR_AndAlsoOp : CIR_Op<"andalso", [NoSideEffect]> {
  let summary = "Performs a short-circuiting boolean AND operation";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
  }];
}

def CIR_OrOp : CIR_Op<"or", [NoSideEffect]> {
  let summary = "Performs a boolean OR operation";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
  }];
}

def CIR_OrElseOp : CIR_Op<"orelse", [NoSideEffect]> {
  let summary = "Performs a short-circuiting boolean OR operation";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
  }];
}

def CIR_XorOp : CIR_Op<"xor", [NoSideEffect]> {
  let summary = "Performs a boolean XOR operation";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
  }];
}

def CIR_NotOp : CIR_Op<"not", [NoSideEffect]> {
  let summary = "Performs a boolean NOT operation";
  let arguments = (ins I1:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value attr-dict
  }];
}

def CIR_IsNullOp : CIR_Op<"is_null", [NoSideEffect]> {
  let summary = "Returns true if the input value is equivalent to NULL";

  let arguments = (ins AnyType:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];
}

def CIR_TypeOfImmediateOp : CIR_Op<"typeof.imm", [NoSideEffect]> {
  let summary = "Returns the type tag of the given immediate";
  let description = [{
    This operation is used to examine the type of an immediate term value, returning
    its primitive type tag as a raw integer value for dispatch.

    For boxed values, this op will return the type of the box itself, i.e.
    TermKind::Box or TermKind::List. Boxed literals are treated as boxes of the
    appropriate type.
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs Index:$kind);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];
}

def CIR_TypeOfBoxOp : CIR_Op<"typeof.box", [MemoryEffects<[MemRead]>]> {
  let summary = "Returns the type tag and arity of the given boxed value";
  let description = [{
    This operation is used to extract data from the header of a boxed term.

    It is assumed that the value given to this operation is guaranteed to be
    of type TermKind::Box. It is unsafe, and undefined behavior, to pass this
    op a value which has not been first checked with TypeOfImmediate.

    This op unwraps the box pointer and dereferences the term header it points to,
    then extracts the kind and arity stored there, returning them for use.
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs Index:$kind, Index:$arity);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];
}

def CIR_TypeOfOp : CIR_Op<"typeof", [MemoryEffects<[MemRead]>]> {
  let summary = "Returns the type information of the given term value";
  let description = [{
    This operation builds on top of TypeOfImmediate and TypeOfBox to obtain the
    primitive type tag for the given term; and the arity of the term, i.e.
    size in words;
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs Index:$kind, Index:$arity);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];
}

def CIR_IsTypeOp : CIR_Op<"is_type", [MemoryEffects<[MemRead]>]> {
  let summary = "A type checking predicate.";
  let description = [{
    This operation is a high-level type check op which returns true if the given term
    is of the same type/kind as the given type. It builds on top of the lower-level
    type checking ops, e.g. TypeOf/TypeOfImmediate/TypeOfBox.

    If you expect to do a chain of type checks, you should prefer to compose the lower level
    ops to avoid redundantly extracting type information from the given term. This op is
    meant to represent one-off type checks
  }];

  let arguments = (ins CIR_AnyTerm:$value, TypeAttr:$expected);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $expected `,` $value `:` type($value) attr-dict
  }];

  //let hasFolder = 1;

  let extraClassDeclaration = [{
    Type getExpectedType() {
      return expected();
    }
  }];
}

def CIR_IsTaggedTupleOp : CIR_Op<"is_tagged_tuple", [MemoryEffects<[MemRead]>]> {
  let summary = "A type checking predicate for records.";
  let description = [{
    This operation is like IsType, but goes further by checking that the tuple
    has an arity of at least 1, and that the first element contains the given atom.
  }];

  let arguments = (ins CIR_AnyTerm:$value, AtomAttr:$tag);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $tag `,` $value `:` type($value) attr-dict
  }];

  //let hasFolder = 1;
}

def CIR_MallocOp : CIR_Op<"malloc", [DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>, MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocates space for a type on the Erlang process heap.";
  let description = [{
    This operation uses the provided type to allocate enough space on
    the process heap to store a value of that type. While this accepts
    any type, only boxable types should be allocated, as immediates require
    no allocation.
  }];

  let arguments = (ins TypeAttr:$allocType);
  let results = (outs PtrOf<CIR_AnyTerm>:$result);

  let assemblyFormat = [{
    $allocType `:` type($result) attr-dict
  }];
}

def CIR_CaptureFunOp : CIR_Op<"fun.capture", [DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>, MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Creates a closure representing the given fun type.";
  let description = [{
    This operation creates a closure object representing the given fun type,
    and using the provided values to fill the free variables of the closure env.
  }];

  let arguments = (ins TypeAttr:$funType, Variadic<CIR_AnyTerm>:$operands);
  let results = (outs CIR_BoxOf<CIR_FunType>);

  let assemblyFormat = [{
    $funType `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def CIR_ConsOp : CIR_Op<"cons", [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "A type constructor for cons cells.";
  let description = [{
    This operation allocates a new cons cell on the process heap
    and writes the given head/tail values to it.
  }];

  let arguments = (ins CIR_AnyTerm:$head, CIR_AnyTerm:$tail);
  let results = (outs CIR_ListType:$result);

  let assemblyFormat = [{
    $head `,` $tail attr-dict `:` `(` type($head) `,` type($tail) `)` `->` type($result)
  }];
}

def CIR_HeadOp : CIR_Op<"hd", [MemoryEffects<[MemRead]>]> {
  let summary = "A cons cell primitive which reads the first element of the cell.";
  let description = [{
    This operation reads the head element from the given boxed cons cell.
  }];

  let arguments = (ins CIR_ListType:$cell);
  let results = (outs CIR_OpaqueTermType:$result);

  let assemblyFormat = [{
    $cell attr-dict
  }];
}

def CIR_TailOp : CIR_Op<"tl", [MemoryEffects<[MemRead]>]> {
  let summary = "A cons cell primitive which reads the last element of the cell.";
  let description = [{
    This operation reads the tail element from the given boxed cons cell.
  }];

  let arguments = (ins CIR_ListType:$cell);
  let results = (outs CIR_OpaqueTermType:$result);

  let assemblyFormat = [{
    $cell attr-dict `:` type($result)
  }];
}

def CIR_TupleOp : CIR_Op<"tuple", [MemoryEffects<[MemAlloc]>]> {
  let summary = "A type constructor for tuples.";
  let description = [{
    This operation allocates a new tuple of the given arity on the process heap.
  }];

  let arguments = (ins I32Attr:$arity);
  let results = (outs CIR_BoxOf<AnyTuple>:$result);

  let assemblyFormat = [{
    `(` $arity `)` attr-dict `:` type($result)
  }];
}

def CIR_SetElementOp : CIR_Op<"setelement", [MemoryEffects<[MemWrite]>]> {
  let summary = "A tuple primitive used to mutate contents of the tuple.";
  let description = [{
    This operation inserts an element in the given tuple at the specified index
  }];

  let arguments = (ins CIR_BoxOf<AnyTuple>:$tuple, Index:$index, CIR_AnyTerm:$value);
  let results = (outs CIR_BoxOf<AnyTuple>:$result);

  let assemblyFormat = [{
    $tuple `[` $index `]` `,` $value attr-dict `:` `(` type($tuple) `,` type($value) `)` `->` type($result)
  }];
}

def CIR_GetElementOp : CIR_Op<"getelement", [MemoryEffects<[MemRead]>]> {
  let summary = "A tuple primitive used to read contents of the tuple.";
  let description = [{
    This operation fetches an element in the given tuple at the specified index
  }];

  let arguments = (ins CIR_BoxOf<AnyTuple>:$tuple, Index:$index);
  let results = (outs CIR_OpaqueTermType:$result);

  let assemblyFormat = [{
    $tuple `[` $index `]` attr-dict `:` `(` type($tuple) `)` `->` type($result)
  }];
}

/*
def CIR_MapOp : CIR_Op<"map", [MemAlloc]> {
  let summary = "A type constructor for maps.";
  let description = [{
    This operation allocates a new empty map.
  }];

  let results = (outs CIR_BoxOf<CIR_Map>:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def CIR_MapPutOp : CIR_Op<"map.put", [MemRead, MemWrite]> {
  let summary = "A map primitive that inserts/updates a value in the map";
  let description = [{
    This operation puts the given value in a map at the provided key, overriding the existing value,
    if present.
  }];

  let arguments = (ins CIR_BoxOf<CIR_Map>:$map, CIR_AnyTerm:$key, CIR_AnyTerm:$value);
  let results = (outs CIR_BoxOf<CIR_Map>:$result);

  let assemblyFormat = [{
    $map `[` $key `]`, $value attr-dict `:` type($result)
  }];
}

def CIR_MapUpdateOp : CIR_Op<"map.update", [MemRead, MemWrite]> {
  let summary = "A map primitive that updates a value in the map";
  let description = [{
    This operation updates the given value in a map at the provided key, raising an exception,
    if the key is not present in the map.
  }];

  let arguments = (ins CIR_BoxOf<CIR_Map>:$map, CIR_AnyTerm:$key, CIR_AnyTerm:$value);
  let results = (outs CIR_BoxOf<CIR_Map>:$result);

  let assemblyFormat = [{
    $map `[` $key `]`, $value attr-dict `:` type($result)
  }];
}
*/

def CIR_RaiseOp : CIR_Op<"raise", [MemoryEffects<[MemRead, MemWrite]>, ReturnLike, Terminator]> {
  let summary = "An exception handling intrinsic that causes a new exception to be raised.";
  let description = [{
    This operation triggers unwinding to the nearest exception landing pad using
    the provided class, reason, and stack trace for the exception details.
  }];

  let arguments = (ins CIR_AtomType:$exceptionClass, CIR_AnyTerm:$exceptionReason, CIR_TraceType:$exceptionTrace);

  let assemblyFormat = [{
    $exceptionClass `,` $exceptionReason `:` type($exceptionReason) `,` $exceptionTrace attr-dict
  }];
}

def CIR_BuildStacktraceOp : CIR_Op<"stacktrace.build", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "An exception handling intrinsic used to construct a stacktrace at the current program counter.";
  let description = [{
    Semantically, this operation constructs a handle to a trace which can later be reified to its Erlang term form upon request.
  }];

  let arguments = (ins);
  let results = (outs CIR_TraceType:$result);

  let assemblyFormat = [{
    attr-dict
  }];
}

def CIR_ExceptionClassOp : CIR_Op<"exception.class", [MemoryEffects<[MemRead]>]> {
  let summary = "An exception handling intrinsic used to obtain the class of the exception, i.e. error, throw, or exit";
  let description = [{
    This operation exists so as to avoid encoding platform-specific details about how exceptions are representing in higher
    level parts of the compiler.
  }];

  let arguments = (ins PtrOf<CIR_ExceptionType>:$exception);
  let results = (outs CIR_AtomType:$result);

  let assemblyFormat = [{
    $exception attr-dict
  }];
}

def CIR_ExceptionReasonOp : CIR_Op<"exception.reason", [MemoryEffects<[MemRead]>]> {
  let summary = "An exception handling intrinsic used to obtain the reason term from an exception.";
  let description = [{
    This operation exists so as to avoid encoding platform-specific details about how exceptions are representing in higher
    level parts of the compiler.
  }];

  let arguments = (ins PtrOf<CIR_ExceptionType>:$exception);
  let results = (outs CIR_OpaqueTermType:$result);

  let assemblyFormat = [{
    $exception attr-dict
  }];
}

def CIR_ExceptionTraceOp : CIR_Op<"exception.trace", [MemoryEffects<[MemRead]>]> {
  let summary = "An exception handling intrinsic used to reify the stacktrace associated with an exception.";
  let description = [{
    This operation is used as the dual to BuildStacktraceOp, in that it converts the trace handle found in the
    exception payload to an Erlang term which can then be operated on by user code.

    In virtually all cases, the resulting term should be a list of tuples representing frames in the trace; but
    in some cases the trace may be nil, meaning capture wasn't possible, or the data to produce a trace was not
    available.
  }];

  let arguments = (ins PtrOf<CIR_ExceptionType>:$exception);
  let results = (outs CIR_OpaqueTermType:$result);

  let assemblyFormat = [{
    $exception attr-dict
  }];
}

def CIR_YieldOp : CIR_Op<"process.yield", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Transfers control flow back to the core scheduler loop.";
  let description = [{
    This operation represents the point in a function at which execution switches back
    to the core scheduler loop or another process. It has different semantics depending
    on the target architecture.

    For example, on non-Wasm architectures, we implement task switching by first saving the
    return address and all callee-saved registers; then restoring the return address and
    callee-saved registers from the state of the scheduler process, at which point a return
    will return back to the yield point and resume execution as if the yield never happened.
    From the point of view of the yielding process, it is a normal function call. The nice
    thing about this is we don't have to do anything special when compiling code for these
    targets as all of the task switching internals are implemented in the runtime.

    However, on Wasm specifically, this approach is not supported due to limitations of the
    platform (as of Spring 2022). Instead, we post-process functions containing yield points
    such that the containing function is split into multiple parts. Each of these parts represents
    a resumption point which corresponds to the yield point from which it was derived. These
    resumption points are set up to receive a continuation object containing the state of the
    function at the yield point. When yielding, that continuation object is allocated and pushed
    on a stack, and then control returns to the caller. When control returns all the way to the scheduler,
    the result is a stack of continuations which can be used to 'rewind' control back to the appropriate
    resumption point.

    To resume from the scheduler, the first continuation pointer is popped from the stack, and
    its function pointer is invoked, passing the continuation pointer as argument. Each resumption
    point examines the continuation it receives to determine whether there is another continuation
    to be popped and followed; or if control should resume in the current function. If the latter,
    all of the values required are loaded from the continuation frame. Continuation frames are discarded
    when control returns normally from a function, but preserved otherwise, so as to be reused when
    control yields again.
  }];

  let assemblyFormat = [{ attr-dict }];
}

def CIR_RecvStartOp : CIR_Op<"recv.start", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Initializes the context necessary to run a receive state machine";
  let description = [{
    This operation sets up the context used by the receive state machine. Aside from
    setting up this state, it performs no other actions.
  }];
  let arguments = (ins CIR_AnyTerm:$timeout);
  let results = (outs CIR_RecvContextType:$context);

  let assemblyFormat = [{
    $timeout `:` type($timeout) attr-dict
  }];
}

def CIR_RecvNextOp : CIR_Op<"recv.next", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Accesses the next message in the process mailbox, if available, and determines what state to transition to";
  let description = [{
    This operation is responsible for determining if there is a message available to
    receive, and if not, whether to time out or yield to the scheduler until either a message
    is received or a timeout occurs.
  }];

  let arguments = (ins PtrOf<CIR_RecvContextType>:$context);
  let results = (outs I8:$state);

  let assemblyFormat = [{
    $context attr-dict
  }];
}

def CIR_RecvPeekOp : CIR_Op<"recv.peek", [MemoryEffects<[MemRead]>]> {
  let summary = "Peeks the current message being inspected by the receive";
  let description = [{
    This operation occurs immediately after a `recv.next` which returns a state indicating that
    a message has been received and is ready to peek. It compiles down to a dereference of the
    message state contained in the receive context.
  }];

  let arguments = (ins PtrOf<CIR_RecvContextType>:$context);
  let results = (outs CIR_OpaqueTermType:$message);

  let assemblyFormat = [{
    $context attr-dict
  }];
}

def CIR_RecvPopOp : CIR_Op<"recv.pop", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Removes the last message peeked from the mailbox";
  let description = [{
    This operation occurs immediately after a `recv.peek` which matches the peeked message.
    The result is that the peeked message is removed from the mailbox.
  }];

  let arguments = (ins PtrOf<CIR_RecvContextType>:$context);
  let results = (outs);

  let assemblyFormat = [{
    $context attr-dict
  }];
}

def CIR_RecvDoneOp : CIR_Op<"recv.done", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Cleans up the receive context";
  let description = [{
    This operation occurs on control flow edges which exit the receive state machine,
    and ensures that the receive context is properly cleaned up. After this instruction
    returns, the context pointer should no longer be used.
  }];

  let arguments = (ins PtrOf<CIR_RecvContextType>:$context);
  let results = (outs);

  let assemblyFormat = [{
    $context attr-dict
  }];
}

def CIR_BinaryStartOp : CIR_Op<"binary.start", [MemoryEffects<[MemWrite]>]> {
  let summary = "Creates a new binary builder";
  let description = [{
    This operation is used to initialize the state used to construct a binary
    from scratch, similar to the kind of context used for receive.
  }];

  let arguments = (ins);
  let results = (outs CIR_BinaryBuilderType:$builder);
  let assemblyFormat = [{
    attr-dict
  }];
}

def CIR_BinaryFinishOp : CIR_Op<"binary.finish", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Completes construction of a new binary";
  let description = [{
    This operation is used to complete construction of a binary, returning the boxed value as a term.

    This operation must always be called on all control flow edges exiting construction of a binary,
    even if the result is discarded.
  }];

  let arguments = (ins CIR_BinaryBuilderType:$builder);
  let results = (outs CIR_BoxOf<CIR_BitsType>:$bin);
  let assemblyFormat = [{
    $builder attr-dict
  }];
}

def CIR_BinaryPushIntegerOp : CIR_Op<"binary.push.int", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Appends an integer value on to the current binary";
  let description = [{
    This operation handles pushing a value of integer type on to the current binary builder, using the provided
    bitsize/flags to determine how the value will be encoded in the resulting binary.
  }];
  let arguments = (ins PtrOf<CIR_BinaryBuilderType>:$builder,
                       CIR_IntegerType:$value,
                       I32:$bits,
                       DefaultValuedAttr<BoolAttr, "false">:$isSigned,
                       DefaultValuedAttr<EndiannessAttr, "::mlir::cir::Endianness::Big">:$endianness,
                       DefaultValuedAttr<I8Attr, "1">:$unit);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def CIR_BinaryPushFloatOp : CIR_Op<"binary.push.float", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Appends a float value on to the current binary";
  let description = [{
    This operation handles pushing a value of float type on to the current binary builder, using the provided
    bitsize/flags to determine how the value will be encoded in the resulting binary.
  }];
  let arguments = (ins PtrOf<CIR_BinaryBuilderType>:$builder,
                       CIR_FloatType:$value,
                       I32:$bits,
                       DefaultValuedAttr<EndiannessAttr, "::mlir::cir::Endianness::Big">:$endianness,
                       DefaultValuedAttr<I8Attr, "1">:$unit);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def CIR_BinaryPushUtf8Op : CIR_Op<"binary.push.utf8", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Appends a utf-8 codepoint value on to the current binary";
  let description = [{
    This operation handles pushing a single utf-8 codepoint on to the current binary builder.
  }];
  let arguments = (ins PtrOf<CIR_BinaryBuilderType>:$builder, CIR_IntegerType:$value);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def CIR_BinaryPushUtf16Op : CIR_Op<"binary.push.utf16", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Appends a utf-16 codepoint value on to the current binary";
  let description = [{
    This operation handles pushing a single utf-16 codepoint on to the current binary builder, using the
    provided flags to determine how the value will be encoded in the resulting binary.
  }];
  let arguments = (ins PtrOf<CIR_BinaryBuilderType>:$builder,
                       CIR_IntegerType:$value,
                       DefaultValuedAttr<EndiannessAttr, "::mlir::cir::Endianness::Big">:$endianness);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def CIR_BinaryPushBitsOp : CIR_Op<"binary.push.bits", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Appends a binary/bitstring value on to the current binary";
  let description = [{
    This operation handles pushing some/all bits of a given binary/bitstring value on to the current binary builder,
    depending on the provided size and unit.

    If a size is given, then only `size * unit` bits will be pushed from the given value. If the size is
    larger than the number of bits in the input value, an error will be raised. If no size is provided, then
    the input is expected to uphold `total_size_in_bits % unit == 0`, otherwise an error will be raised.
  }];

  let arguments = (ins PtrOf<CIR_BinaryBuilderType>:$builder,
                       CIR_BoxOf<CIR_BitsType>:$value,
                       CIR_IntegerType:$unit,
                       Optional<CIR_IntegerType>:$size);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

#endif