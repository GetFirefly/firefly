#ifndef EIR_DIALECT_H
#define EIR_DIALECT_H

#include "mlir/IR/Dialect.h"

using ::llvm::StringRef;

namespace lumen {
namespace eir {

/// This is the definition of the EIR dialect.
///
/// This is responsible for registering all of the custom operations,
/// types, and attributes that are unique to EIR within the MLIR syntax
class EirDialect : public mlir::Dialect {
public:
  explicit EirDialect(mlir::MLIRContext *ctx);

  /// Parse an instance of a type registered to this dialect
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  /// Print an instance of a type registered to this dialect
  void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const override;

  /// Print an instance of an attribute registered to this dialect
  void printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const override;

  /// Provide a utility accessor to the dialect namespace.
  ///
  /// This is used by several utilities for casting between dialects.
  static StringRef getDialectNamespace() { return "eir"; }
};

} // namespace eir
} // namespace lumen

#endif // EIR_DIALECT_H
