#ifndef EIR_DIALECT_TD
#define EIR_DIALECT_TD

include "mlir/IR/OpBase.td"

// We need to define the EIR dialect within the ODS framework,
// so that our operations can be defined against it
def eir_Dialect : Dialect {
  let name = "eir";
  let summary = "An MLIR dialect that corresponds to EIR produced by the Lumen front-end";
  let cppNamespace = "eir";
}

#endif
