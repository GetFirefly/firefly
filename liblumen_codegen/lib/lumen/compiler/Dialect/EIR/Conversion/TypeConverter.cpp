#include "lumen/compiler/Dialect/EIR/Conversion/TypeConverter.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"

#include "mlir/IR/StandardTypes.h"

using ::mlir::Type;

using namespace ::lumen::eir;

static bool isa_eir_type(Type t);
static bool isa_std_type(Type t);

StandardTypeConverter::StandardTypeConverter(TargetInfo &ti)
  : TypeConverter(), targetInfo(ti) {
  addConversion([this](Type t) -> Optional<Type> {
      if (isa_std_type(t))
        return t;

      if (!isa_eir_type(t))
        return Optional<Type>();

      MLIRContext *context = t.getContext();
      OpaqueTermType termTy = t.cast<OpaqueTermType>();
      if (termTy.isImmediate() && !termTy.isBox())
        return mlir::IntegerType::get(targetInfo.pointerSizeInBits, context);

      t.dump();
      assert(false && "unimplemented conversion");

      return llvm::None;
  });
}


static bool isa_eir_type(Type t) {
  return inbounds(t.getKind(),
                  Type::Kind::FIRST_EIR_TYPE,
                  Type::Kind::LAST_EIR_TYPE);
}


static bool isa_std_type(Type t) {
  return inbounds(t.getKind(),
                  Type::Kind::FIRST_STANDARD_TYPE,
                  Type::Kind::LAST_STANDARD_TYPE);
}
