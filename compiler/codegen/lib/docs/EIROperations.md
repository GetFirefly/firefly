# EIR Operations

The following are all of the various operations that are defined by the EIR
dialect.

## Calls

### call

    eir.call @fn(%arg1, .., %argN) : fnTy

### indirect_call

    eir.indirect_call %fn(%arg1, .., %argN) : fnTy

NOTE: Indirect calls may not always know the return type, but for Erlang
applications we know that any call must result in either termination of
the process, or returning a Term. In such cases where the callee is unknown,
we assume that the return type is thus `!eir.term`

## Control Flow

### br

An unconditional branch instruction. This is a terminator operation.

    eir.br ^block(...)
    
This basically mirrors the standard `br` operation, but is kept in EIR dialect
to differentiate between control flow from EIR, and control flow introduced by
code generation.

Primarily meant for lowering EIR calls which refer to a block for control flow,
rather than a function.

### if_bool

A conditional branching construct. This is a terminator operation.

    eir.if_bool %selector, ^true(...), ^false(...), ^else(...)
    
If `^else` is not provided, then it is assumed that such a branch is unreachable.

### if_type

A conditional branch instruction based on whether the type of the selector
matches the specified type. This is a terminator operation.

    eir.if_type<"T"> %selector, ^true(...), ^false(...)
    
Syntactically, this operates in the same fashion as `eir.is_type`. See the notes
about that operation for details on type conversion rules.

If the selector matches the given type, control branches to the `^true` block,
and the selector is passed as the first argument, followed by any specified
arguments. Importantly, the selector is also cast to the specified type before
passed to the successor block, if necessary. This allows one to write optimized
paths for cases where types match.

If the selector does not match, control branches to the `^false` block, and the
selector is passed as the first argument, followed by any specified arguments.
The selector is passed along as-is in this case.

### match

A richer and more dynamic form of conditional branching. Like `if_bool`, this is
a terminator operation.

    eir.match %selector, (branch to ^block(...))+
    
NOTE: Block arguments may be given and will be appended after any arguments
resulting from the type of match.

`branch` is defined as:


    branch ::= match-kind
    match-kind ::= value-match
                | type-match
                | wildcard-match
                | tuple-match
                | cons-match
                | map-entry-match
                | binary-match
    value-match     ::= 'value'(ssa-id)
    type-match      ::= 'type'(type-id)
    wildcard-match  ::= 'any'
    tuple-match     ::= 'tuple'(arity:integer-literal)
    cons-match      ::= 'cons'
    map-entry-match ::= 'map_entry'(key:ssa-id, value:ssa-id)
    binary-match    ::= 'binary'(size:ssa-id, selector+)
                     |  'binary'(selector+)
    selector ::= unit-type
              | signedness
              | endianness
              | 'unit'(integer-literal)
    unit-type ::= 'integer'
               | 'float'
               | 'bytes'
               | 'bits'
               | 'utf8'
               | 'utf16'
               | 'utf32'
    signedness ::= 'signed' | 'unsigned'
    endianness ::= 'native' | 'big' | 'little'


Some restrictions/addendums to the above apply:

- The `size` argument to the binary match type may not be given if any of the
  `utf` unit types are given. They are incomptabile.
- The signedness defaults to `unsigned`, and can only be given if `integer` is
  given
- The endianness defaults to `big`, and can only be given if one of `integer`, `float`,
  `utf16`, or `utf32` unit types are given.
- `unit` cannot be given if any `utf` unit type is given
- If `size` and `unit` is omitted, then:
  - unit is assumed to be 1 for `integer`, `float` and `bits`, 8 for `bytes`
  - size is assumed to be 8 for `integer`, 64 for `float` and the entire
    binary/bitstring for `bytes`/`bits` respectively
    
## Error Handling

### capture_trace

Captures the current stack trace and returns a reference to it as an SSA value

    %trace_ref = eir.capture_trace
    
### get_trace

Access the stack trace previously captured by `capture_trace`, and returns it as
an SSA value.

    %trace_ref = eir.get_trace


## Data

### box

Given a reference, produces a boxed value that can be passed to the runtime.

    %boxed = eir.box(%val : !eir.ref<T>) : !eir.box<T>
    
### unbox

The inverse of `box`, extracts a reference contained in the boxed value.

    %ref = eir.unbox(%val : !eir.box<T>) : !eir.ref<T>

### is_type

A dynamic type checking operation.

    %result : i1 = eir.is_type<"T"> %selector
    
In the above, `T` is a placeholder for a type signature. While this is really
intended specifically for EIR types, it is possible to specify arbitrary types
here, but during lowering, an assertion will be thrown if there is no supported
conversion to an EIR type. See the `eir.cast` notes for more details.

If you want to branch on type, and use the typed value, use `eir.if_type` instead, as it
combines type checking and casting the value in one combined operation, which is
more efficient when the typed value is needed. This operation is intended as a
more primitive operation consumed by other operations like `match`, where type
checking is required, but the value is not required to be cast to the matched type.

### cast

A dynamic cast operation.

    %result = eir.cast %value:inTy to outTy
    
This operation performs a cast from `inTy` to `outTy`, if supported. In most
cases, this will be used to cast an opaque type to a more concrete type, i.e.
converting `eir.term` to `eir.fixnum` or `eir.box<T>`. However, it may also be
used to convert from EIR types to standard types, e.g. `eir.fixnum` to `i32`.
Some casts are not supported, such as an EIR type to memref. It is also not
supported to cast between other dialect types; an EIR type must be either the
source or target type.

The various supported implicit casts are below:

- `iX` from `eir.fixnum`, where X is the bit width the value must fit in
  - this is done by stripping the tag bits from the fixnum
- `eir.fixnum` from `iX`, as long as X is less than or equal to the bit width of
  the maximum fixnum width.
  - this is done by sign-extending the iX, clearing the tag bits or shifting,
    then adding the fixnum tag
- `i1` from `eir.boolean`
  - this is done by stripping the tag bits from the boolean atom; 'false' is 0,
    and 'true' is 1
- `eir.boolean` from `i1`
  - this is done by zero-extending the i1, and adding the atom tag
- `index` from `eir.fixnum` when the value is unsigned and fits in a
  pointer-width integer
- `eir.fixnum` from `index` when the value fits in the fixnum width
- `f64` from `eir.float`
  - on x86_64, this is just a bitcast
  - on other platforms, we don't use `eir.float`
- `eir.float` from `f64`
  - per the above, this is just a bitcast
- `f64` from `eir.float.packed`
  - this is done by loading the f64 directly from the boxed term
  - once unpacked, we do not repack unless necessary
  - NOTE: we do _not_ support conversions to `eir.float.packed`, instead a new
    packed float must be allocated/constructed from the `f64` value
- `eir.ref<T>` from `eir.box<T>`
  - i.e. extract the pointer from the box, this delegates to the `eir.unbox` operation
- `eir.box<T>` from `eir.ref<T>`
  - i.e. box the pointer, this delegates to the `eir.box` operation
- `vector<NxT>` for `eir.tuple` of `N` arity, where `T` is a word-sized EIR type
  - this is essentially a bitcast, which takes the address of the first element
    of the tuple and uses that as the base address for the vector
  - this conversion takes advantage of the fact that our tuple layout and a
    vector layout are 1:1, but this only holds when the vector elements are all
    word-sized, i.e. `eir.term`, `eir.fixnum`, `eir.boolean`, `eir.atom`,
    `eir.nil`, and the other immediates.

This operation is a primitive meant to be used by higher-level operations, i.e.
`if_type` will check the type with `is_type`, cast with `cast`, then branch with
`br`. Later passes will then optimize series of instructions like these to avoid
any duplicate work they may do internally.

### map_insert

Inserts a key/value pair into the given map, and jumps to either the success or
failure block based on the result of the operation. This is a terminator operation.

    eir.map_insert(%map, %key, %value, ^success(...), ^failure(...))
    
NOTE: The updated map will be given as the first block argument to `^success`.
Failure of the operation can occur if `%map` is not a map term, or if `%key`
already exists in `%map`.

### map_update

Has almost the same semantics as `map_insert`, but rather than expecting the key
to not exist, it expects the key to exist.

    eir.map_update(%map, %key, %value, ^success(...), ^failure(...))
    
### map_get

Extracts the value with the given key from a map.

    %value = eir.map_get(%map, %key)
    
NOTE: Can only be used when `%map` is known to be a map and `%key` is a term.
    
### extract_element

Extracts an element from a tuple, by zero-based index.

Think of this as a higher-level version of LLVM's `extractelement`.

    %element = eir.extract_element(%tuple, %index)
    
NOTE: Can only be used when `%tuple` is known to be a tuple and `%index` is an
integer value.

### update_element

Destructively updates (i.e. in-place) an element in a tuple, by zero-based index.

Think of this as a higher-level version of LLVM's `insertelement`.

    eir.update_element(%tuple, %index, %value)
    
NOTE: Can only be used when `%tuple` is known to be a tuple and `%index` is an
integer value; in addition, `%value` must be a valid word-sized EIR type.

### head

Like `extract_element`, but for cons cells, and specifically extracts the head
element.

    %head = eir.head(%cons)
    
NOTE: Can only be used when `%cons` is known to be a cons cell.
    
### tail

Like `extract_element`, but for cons cells, and specifically extracts the tail
element.

    %tail = eir.tail(%cons)
    
NOTE: Can only be used when `%cons` is known to be a cons cell.

### binary_push

Used during binary construction, this operation appends a new value to the given
binary, using the given binary specifier. Once complete, control branches to
either the success or failure block based on the result of the operation. This
is a terminator operation.

    eir.binary_push(%bin, %val, binary-spec, ^success(...), ^failure(...))
    
Where `binary-spec` is defined as:

    binary-spec ::= binary(size:ssa-id, specifier+)
                 |  binary(specifier+)
                 
    specifier ::= see definition of `branch` in `eir.match` operation
    
NOTE: This uses the same specifier syntax as `eir.match`. In the success case,
the new binary is passed as a block argument to the success block. Failure can
occur if `%bin` is not a binary term, `%val` is invalid for the given binary
spec, or due to allocation failure (either on heap or stack).


## Miscellaneous

### unreachable

Semantically identical to the standard `unreachable`, but is preserved for our
own analyses in case we later decide to handle them differently.

    eir.unreachable
   
## Intrinsics

### intrinsic<"name">

A generalized intrinsic, which unless otherwise handled during lowering, will be
lowered as a function call to a runtime function of the same name.

Syntactically, this takes essentially the same form as a function call:

    %ret = eir.intrinsic<"name">(%arg1, .., %argN) : fnTy
