#ifndef LUMEN_COMPILER_DIALECT_EIR_CONVERSION_TYPECONVERTER_H_
#define LUMEN_COMPILER_DIALECT_EIR_CONVERSION_TYPECONVERTER_H_

#include "lumen/compiler/Target/TargetInfo.h"

#include "mlir/Transforms/DialectConversion.h"

namespace lumen {
namespace eir {

/// Converts EIR types to Standard types
class StandardTypeConverter final : public mlir::TypeConverter {
 public:
  using TypeConverter::TypeConverter;

  StandardTypeConverter(TargetInfo &targetInfo)
      : TypeConverter(), targetInfo(targetInfo) {}

  /// Converts the given EIR `type` to Standard
  mlir::Type convertType(mlir::Type type) override;

private:
  TargetInfo targetInfo;
};

}  // namespace eir
}  // namespace lumen

#endif
