#ifndef CIR_TYPE_INTERFACES_TD
#define CIR_TYPE_INTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Type Interfaces
//===----------------------------------------------------------------------===//

def BoxedTypeInterface : TypeInterface<"BoxedType"> {
  let cppNamespace = "::mlir::cir";
  let description = [{
    This type interface indicates that the implementing type cannot be used unboxed,
    i.e. it must always be allocated on the process heap and passed by-reference.
  }];
  let methods = [
    InterfaceMethod<[{
      Returns the type tag for the underlying type
    }],
    "::lumen::TermKind::Kind", "getTermKind">,
  ];
}

def ImmediateTypeInterface : TypeInterface<"ImmediateType"> {
  let cppNamespace = "::mlir::cir";
  let description = [{
    This type interface indicates that the implementing type is valid as an immediate,
    i.e. it is trivially movable and is no larger than the machine word-size
  }];
  let methods = [
    InterfaceMethod<[{
      Returns the type tag for the underlying type
    }],
    "::lumen::TermKind::Kind", "getTermKind">,
  ];
}

def TargetSensitiveReprTypeInterface : TypeInterface<"TargetSensitiveReprType"> {
  let cppNamespace = "::mlir::cir";
  let description = [{
    Some Erlang terms have representations that are immediate on some targets but
    boxed on others. This interface is used as a fallback when checking whether a
    type is immediate or boxable, by providing the target encoding which makes the
    choice of representation concrete.
  }];
  let methods = [
    InterfaceMethod<[{
      Returns the type tag for the underlying type
    }],
    "::lumen::TermKind::Kind", "getTermKind">,
    InterfaceMethod<[{
      Returns true if the type is immediate in the given encoding
    }],
    "bool", "isImmediate", (ins "const ::lumen::Encoding &":$encoding)>
  ];
}

#endif
