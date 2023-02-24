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
    $module attr-dict-with-keyword $region
  }];

  let hasRegionVerifier = 1;

  let extraClassDeclaration = [{
    void appendTableEntry(Operation *op);

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
    OpBuilder<(ins "func::FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
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

def CIR_EnterOp : CIR_Op<"enter",
    [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>, ReturnLike, Terminator]> {
  let summary = "tail call operation";
  let description = [{
    This operation is equivalent to `cir.call` except that it acts as a block terminator,
    and it must always occur in tail position.

    Example:

    ```mlir
    cir.enter @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs);

  let builders = [
    OpBuilder<(ins "func::FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
    }]>
  ];

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
    $callee `(` $operands `)` attr-dict `:` type($operands)
  }];
}

def CIR_CallIndirectOp : CIR_Op<"call.indirect", [CallOpInterface]> {
  let summary = "indirect call operation to a known fun";
  let description = [{
    This operation represents an indirect call to a value that we know is a valid
    fun created by erlang:make_fun/3, allowing us to bypass the overhead of a full erlang:apply/3
    invocation.

    NOTE: The calling convention of the callee _must_ be Erlang, and this is encoded in
    the result type of this operation.
  }];

  let arguments = (ins CIR_BoxOf<CIR_FunType>:$callee, Variadic<AnyType>:$callee_operands);
  let results = (outs I1:$is_err, CIR_OpaqueTermType:$result);

  let builders = [
    OpBuilder<(ins "Value":$callee, CArg<"ValueRange", "{}">:$args), [{
      $_state.operands.push_back(callee);
      $_state.addOperands(args);
      auto i1Ty = $_builder.getI1Type();
      auto termTy = CIROpaqueTermType::get($_builder.getContext());
      $_state.addTypes({i1Ty, termTy});
    }]>,
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    ValueRange operands() { return getCalleeOperands(); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }
  }];

  //let hasCanonicalizeMethod = 1;
  let assemblyFormat = [{
    $callee `(` $callee_operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def CIR_EnterIndirectOp : CIR_Op<"enter.indirect", [CallOpInterface, ReturnLike, Terminator]> {
  let summary = "indirect tail call operation to a known fun";
  let description = [{
    This operation is equivalent to `call.indirect`, except that it acts as a terminator, and must
    occur in tail position.
  }];

  let arguments = (ins CIR_BoxOf<CIR_FunType>:$callee, Variadic<AnyType>:$callee_operands);
  let results = (outs);

  let builders = [
    OpBuilder<(ins "Value":$callee, CArg<"ValueRange", "{}">:$args), [{
      $_state.operands.push_back(callee);
      $_state.addOperands(args);
    }]>,
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    ValueRange operands() { return getCalleeOperands(); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }
  }];

  //let hasCanonicalizeMethod = 1;
  let assemblyFormat = [{
    $callee `(` $callee_operands `)` attr-dict `:` type(operands)
  }];
}

def CIR_CastOp : CIR_Op<"cast",
  [DeclareOpInterfaceMethods<CastOpInterface>, Pure]> {
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
  [ConstantLike, Pure]> {
    let summary = "constant or immediate term";
    let description = [{
      The `constant` operation produces an SSA value equal to some term constant specified by an attribute.
    }];

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$result);

    let builders = [
      OpBuilder<(ins "Attribute":$value, "Type":$type),
      [{ build($_builder, $_state, type, value); }]>,
    ];

    let extraClassDeclaration = [{
      // Returns true if a constant can be constructed with the given value and type
      static bool isBuildableWith(Attribute value, Type type);
    }];

    let hasFolder = 1;
    let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";
}

def CIR_ConstantNullOp : CIR_Op<"null", [Pure]> {
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

def CIR_AndOp : CIR_Op<"and", [Pure]> {
  let summary = "Performs a boolean AND operation";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
  }];
}

def CIR_AndAlsoOp : CIR_Op<"andalso", [Pure]> {
  let summary = "Performs a short-circuiting boolean AND operation";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
  }];
}

def CIR_OrOp : CIR_Op<"or", [Pure]> {
  let summary = "Performs a boolean OR operation";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
  }];
}

def CIR_OrElseOp : CIR_Op<"orelse", [Pure]> {
  let summary = "Performs a short-circuiting boolean OR operation";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
  }];
}

def CIR_XorOp : CIR_Op<"xor", [Pure]> {
  let summary = "Performs a boolean XOR operation";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
  }];
}

def CIR_NotOp : CIR_Op<"not", [Pure]> {
  let summary = "Performs a boolean NOT operation";
  let arguments = (ins I1:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value attr-dict
  }];
}

def CIR_IsNullOp : CIR_Op<"is_null", [Pure]> {
  let summary = "Returns true if the input value is equivalent to NULL";

  let arguments = (ins AnyType:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];
}

def CIR_TruncOp : CIR_Op<"trunc", [Pure]> {
  let summary = "Returns the provided integer truncated to the given type bitwidth";

  let arguments = (ins AnyType:$value);
  let results = (outs AnyInteger:$result);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def CIR_ZExtOp : CIR_Op<"zext", [Pure]> {
  let summary = "Returns the provided integer zero-extended to the given type bitwidth";

  let arguments = (ins AnyType:$value);
  let results = (outs AnyInteger:$result);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def CIR_TypeOfOp : CIR_Op<"typeof", [MemoryEffects<[MemRead]>]> {
  let summary = "Returns the type information of the given term value";
  let description = [{
    This operation returns the type code associated with the term type of the given value.
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs I32:$kind);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];
}

def CIR_IsListOp : CIR_Op<"is_list", [MemoryEffects<[MemRead]>]> {
  let summary = "A type checking predicate for lists.";
  let description = [{
    This operation returns true if the given term is nil or a pointer to a cons cell.
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];

  //let hasFolder = 1;
}

def CIR_IsNonEmptyListOp : CIR_Op<"is_nonempty_list", [MemoryEffects<[MemRead]>]> {
  let summary = "A type checking predicate for nonempty lists.";
  let description = [{
    This operation returns true if the given term is a pointer to a cons cell.
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];

  //let hasFolder = 1;
}

def CIR_IsNumberOp : CIR_Op<"is_number", [MemoryEffects<[MemRead]>]> {
  let summary = "A type checking predicate for numbers.";
  let description = [{
    This operation returns true if the given term is any integer type or a float.
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];

  //let hasFolder = 1;
}

def CIR_IsFloatOp : CIR_Op<"is_float", [MemoryEffects<[MemRead]>]> {
  let summary = "A type checking predicate for floats.";
  let description = [{
    This operation returns true if the given term is a float
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];

  //let hasFolder = 1;
}

def CIR_IsIntegerOp : CIR_Op<"is_integer", [MemoryEffects<[MemRead]>]> {
  let summary = "A type checking predicate for integers.";
  let description = [{
    This operation returns true if the given term is any integer type.
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];

  //let hasFolder = 1;
}

def CIR_IsIsizeOp : CIR_Op<"is_isize", [MemoryEffects<[MemRead]>]> {
  let summary = "A type checking predicate for fixed-width integers.";
  let description = [{
    This operation returns true if the given term is a fixed-width integer value
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];

  //let hasFolder = 1;
}

def CIR_IsBigIntOp : CIR_Op<"is_bigint", [MemoryEffects<[MemRead]>]> {
  let summary = "A type checking predicate for big integers.";
  let description = [{
    This operation returns true if the given term is a big integer value
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];

  //let hasFolder = 1;
}

def CIR_IsAtomOp : CIR_Op<"is_atom", [MemoryEffects<[MemRead]>]> {
  let summary = "A type checking predicate for atoms.";
  let description = [{
    This operation returns true if the given term is a valid atom or boolean value.
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];

  //let hasFolder = 1;
}

def CIR_IsBoolOp : CIR_Op<"is_bool", [MemoryEffects<[MemRead]>]> {
  let summary = "A type checking predicate for booleans.";
  let description = [{
    This operation returns true if the given term is a boolean value.
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];

  //let hasFolder = 1;
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
      return getExpected();
    }
  }];
}

def CIR_IsTupleOp : CIR_Op<"is_tuple", [MemoryEffects<[MemRead]>]> {
  let summary = "A type checking predicate for tuples.";
  let description = [{
    This operation determines if the given term is a tuple, and also determines the
    arity of the tuple if it is one.
  }];

  let arguments = (ins CIR_AnyTerm:$value);
  let results = (outs I1:$result, I32:$arity);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];

  //let hasFolder = 1;
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

  let arguments = (ins CIR_ProcessType:$process, TypeAttr:$allocType);
  let results = (outs PtrOf<CIR_AnyTerm>:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict functional-type(operands, results)
  }];
}

def CIR_MakeFunOp : CIR_Op<"fun.make", [DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>, MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Creates a closure representing the given fun type.";
  let description = [{
    This operation creates a closure object representing the given fun type,
    and using the provided values to fill the free variables of the closure env.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, CIR_ProcessType:$process, Variadic<CIR_AnyTerm>:$env);
  let results = (outs I1:$is_err, CIR_BoxOf<CIR_FunType>:$result);

  let builders = [
    OpBuilder<(ins "func::FuncOp":$callee, "Value":$process, CArg<"ValueRange", "{}">:$operands), [{
      auto calleeType = callee.getFunctionType();
      SmallVector<Type, 1> envTypes;
      for (auto operand : operands)
        envTypes.push_back(operand.getType());
      auto funTy = CIRFunType::get($_builder.getContext(), calleeType, envTypes);
      $_state.addOperands({process});
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addAttribute("funType", TypeAttr::get(funTy));
      $_state.addTypes($_builder.getI1Type());
      $_state.addTypes(CIRBoxType::get(funTy));
    }]>,
  ];

  let assemblyFormat = [{
    $callee `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def CIR_UnpackEnvOp : CIR_Op<"fun.env.get", [MemoryEffects<[MemRead]>]> {
  let summary = "A closure primitive used to read contents of the closure environment.";
  let description = [{
    This operation fetches an element in the given closure at the specified index
  }];

  let arguments = (ins CIR_OpaqueTermType:$fun, IndexAttr:$index);
  let results = (outs CIR_OpaqueTermType:$result);

  let assemblyFormat = [{
    $fun `[` $index `]` attr-dict
  }];
}

def CIR_ConsOp : CIR_Op<"cons", [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "A type constructor for cons cells.";
  let description = [{
    This operation allocates a new cons cell on the process heap
    and writes the given head/tail values to it.
  }];

  let arguments = (ins CIR_ProcessType:$process, CIR_AnyTerm:$head, CIR_AnyTerm:$tail);
  let results = (outs CIR_BoxOf<CIR_ConsType>:$result);

  let assemblyFormat = [{
    $process `,` `[` $head `:` type($head) `,` $tail `:` type($tail) `]` attr-dict
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
    $cell attr-dict
  }];
}

def CIR_SetElementOp : CIR_Op<"setelement", [MemoryEffects<[MemWrite]>]> {
  let summary = "A tuple primitive used to mutate contents of the tuple.";
  let description = [{
    This operation inserts an element in the given tuple at the specified index
  }];

  let arguments = (ins CIR_BoxOf<AnyTuple>:$tuple, IndexAttr:$index,
                        CIR_AnyTerm:$value, BoolAttr:$in_place);
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

  let arguments = (ins CIR_BoxOf<AnyTuple>:$tuple, IndexAttr:$index);
  let results = (outs CIR_OpaqueTermType:$result);

  let assemblyFormat = [{
    $tuple `[` $index `]` attr-dict `:` `(` type($tuple) `)` `->` type($result)
  }];
}

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
  let results = (outs PtrOf<CIR_TraceType>:$result);

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

def CIR_BinaryMatchStartOp : CIR_Op<"bs.match.start", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Creates a new binary pattern matching context";
  let description = [{
    This operation is used to initialize the state used to perform pattern
    matching over bitstrings/binaries.

    The input argument must be a binary term.
  }];

  let arguments = (ins CIR_AnyTerm:$bin);
  let results = (outs I1:$is_err, CIR_OpaqueTermType:$result);
  let assemblyFormat = [{
    $bin `:` type($bin) attr-dict
  }];
}

def CIR_BinaryMatchOp : CIR_Op<"bs.match", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Extracts a matching value from the given match context";
  let description = [{
    This operation is used to extract a value matching the given specification/size
    from a match context created by 'bs.match.start'.

    The first argument must be a match context value, the size argument is optional.
  }];

  let arguments = (ins PtrOf<CIR_MatchContextType>:$matchContext, BinarySpecAttr:$spec, Optional<CIR_AnyTerm>:$size);
  let results = (outs I1:$is_err, CIR_OpaqueTermType:$extracted, PtrOf<CIR_MatchContextType>:$updatedMatchContext);
  let assemblyFormat = [{
    $spec `in` $matchContext (`sized` $size^ `:` type($size))? attr-dict
  }];
}

def CIR_BinaryMatchSkipOp : CIR_Op<"bs.match.skip", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Matches a value from the given match context, but skips extraction";
  let description = [{
    This operation largely mirrors `bs.match`, but instead of extracting a term, it simply
    advances the position of the match context if the match was successful. this is an optimization
    for binary matches which do not use the extracted term and so enables us to avoid unused allocations.

    The first argument must be a match context value, the size argument is required, unlike `bs.match`.
  }];

  let arguments = (ins PtrOf<CIR_MatchContextType>:$matchContext, BinarySpecAttr:$spec, CIR_AnyTerm:$size, I64:$value);
  let results = (outs I1:$is_err, PtrOf<CIR_MatchContextType>:$updatedMatchContext);
  let assemblyFormat = [{
    $value `with` $spec `in` $matchContext `sized` $size `:` type($size) attr-dict
  }];
}

def CIR_BinaryTestTailOp : CIR_Op<"bs.test.tail", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Tests that the tail of a binary match context is of the expected size";
  let description = [{
    This operation is used to match the tail of a binary by size.

    The first argument must be a match context value, the size argument must be an integer.
  }];

  let arguments = (ins PtrOf<CIR_MatchContextType>:$matchContext, IndexAttr:$size);
  let results = (outs I1:$is_err);
  let assemblyFormat = [{
    $matchContext `,` $size attr-dict
  }];
}

def CIR_BinaryPushOp : CIR_Op<"bs.push", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Pushes a value on to a bitstring buffer according to a spec and size";
  let description = [{
    This operation is used to push a value on to an initialized bitstring buffer during
    binary construction.

    The first argument must be a binary builder value, and the size argument is optional.
  }];

  let arguments = (ins PtrOf<CIR_BinaryBuilderType>:$bin, BinarySpecAttr:$spec, CIR_AnyTerm:$value, Optional<CIR_AnyTerm>:$size);
  let results = (outs I1:$is_err, CIR_OpaqueTermType:$updated);
  let assemblyFormat = [{
    $spec $value `:` type($value) `onto` $bin (`sized` $size^ `:` type($size))? attr-dict
  }];
}

#endif
