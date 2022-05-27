#ifndef CIR_TYPE_INTERFACES_TD
#define CIR_TYPE_INTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Type Interfaces
//===----------------------------------------------------------------------===//

def TermTypeInterface : TypeInterface<"TermType"> {
  let cppNamespace = "::mlir::cir";
  let description = [{
    This interface must be implemented for all types which are valid term types,
    whether immediate, boxed, or target-sensitive.
  }];
}

def BoxedTypeInterface : TypeInterface<"BoxedType"> {
  let cppNamespace = "::mlir::cir";
  let description = [{
    This type interface indicates that the implementing type cannot be used unboxed,
    i.e. it must always be allocated on the process heap and passed by-reference.
  }];
}

def ImmediateTypeInterface : TypeInterface<"ImmediateType"> {
  let cppNamespace = "::mlir::cir";
  let description = [{
    This type interface indicates that the implementing type is valid as an immediate,
    i.e. it is trivially movable and is no larger than the machine word-size
  }];
}

#endif
