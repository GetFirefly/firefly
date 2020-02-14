#include "lumen/compiler/Dialect/EIR/Conversion/TypeConverter.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"

#include "mlir/IR/StandardTypes.h"

using ::mlir::Type;

using namespace ::lumen::eir;

Type StandardTypeConverter::convertType(Type t) {
  if (isa_std_type(t))
    return t;

  if (!isa_eir_type(t))
    return nullptr;
        
  MLIRContext *context = t.getContext();
  OpaqueTermType termTy = t.cast<OpaqueTermType>();
  if (termTy.isImmediate() && !termTy.isBox())
      return mlir::IntegerType::get(targetInfo.pointerSizeInBits, context);

  t.dump();
  assert(false && "unimplemented conversion");

  return nullptr;
}
