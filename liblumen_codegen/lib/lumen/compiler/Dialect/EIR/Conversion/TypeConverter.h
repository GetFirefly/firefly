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

  StandardTypeConverter(TargetInfo &targetInfo);

private:
  TargetInfo targetInfo;
};

}  // namespace eir
}  // namespace lumen

#endif
