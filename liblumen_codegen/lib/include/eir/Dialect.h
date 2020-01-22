#ifndef EIR_DIALECT_H
#define EIR_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace M = mlir;
namespace L = llvm;

namespace eir {

/// This is the definition of the EIR dialect.
///
/// This is responsible for registering all of the custom operations,
/// types, and attributes that are unique to EIR within the MLIR syntax
class EirDialect : public M::Dialect {
public:
  explicit EirDialect(M::MLIRContext *ctx);

  /// Parse an instance of a type registered to this dialect
  M::Type parseType(M::DialectAsmParser &parser) const override;

  /// Print an instance of a type registered to this dialect
  void printType(M::Type type, M::DialectAsmPrinter &printer) const override;

  /// Provide a utility accessor to the dialect namespace.
  ///
  /// This is used by several utilities for casting between dialects.
  static L::StringRef getDialectNamespace() { return "eir"; }

  /// Materialize a single constant operation from a given attribute value with
  /// the desired resultant type.
  M::Operation *materializeConstant(M::OpBuilder &builder, M::Attribute value, M::Type type,
                                    M::Location loc) override;
};

} // namespace eir

#endif // EIR_DIALECT_H
