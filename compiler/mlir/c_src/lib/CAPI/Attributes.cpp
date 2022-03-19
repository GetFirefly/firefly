#include "CIR-c/Dialects.h"

#include "CIR/Attributes.h"
#include "CIR/Dialect.h"
#include "CIR/Types.h"

#include "mlir/CAPI/IR.h"

using namespace mlir;
using namespace mlir::cir;

MlirAttribute mlirCirAtomAttrGet(AtomRef atom, MlirType ty) {
  Type type = unwrap(ty);
  return wrap(AtomAttr::get(type.getContext(), type, atom));
}

bool mlirCirAtomAttrIsA(MlirAttribute attr) {
  return unwrap(attr).isa<AtomAttr>();
}

AtomRef mlirCirAtomAttrValueOf(MlirAttribute attr) {
  return unwrap(attr).cast<AtomAttr>().getValue();
}

MlirAttribute mlirCirEndiannessAttrGet(CirEndianness value, MlirContext ctx) {
  MLIRContext *context = unwrap(ctx);

  auto ty = IntegerType::get(context, 8);
  return wrap(EndiannessAttr::get(context, ty, unwrap(value)));
}

bool mlirCirEndianessAttrIsA(MlirAttribute attr) {
  return unwrap(attr).isa<EndiannessAttr>();
}

CirEndianness mlirCirEndiannessAttrValueOf(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<EndiannessAttr>().getValue());
}
