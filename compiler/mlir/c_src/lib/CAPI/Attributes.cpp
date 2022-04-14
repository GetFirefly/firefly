#include "CIR-c/Dialects.h"

#include "CIR/Attributes.h"
#include "CIR/Dialect.h"
#include "CIR/Types.h"

#include "mlir/CAPI/IR.h"

using namespace mlir;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
/// NoneAttr
//===----------------------------------------------------------------------===//

MlirAttribute mlirCirNoneAttrGet(MlirContext ctx) {
  return wrap(NoneAttr::get(unwrap(ctx)));
}

bool mlirCirNoneAttrIsA(MlirAttribute attr) {
  return unwrap(attr).isa<NoneAttr>();
}

//===----------------------------------------------------------------------===//
/// NilAttr
//===----------------------------------------------------------------------===//

MlirAttribute mlirCirNilAttrGet(MlirContext ctx) {
  return wrap(NilAttr::get(unwrap(ctx)));
}

bool mlirCirNilAttrIsA(MlirAttribute attr) {
  return unwrap(attr).isa<NilAttr>();
}

//===----------------------------------------------------------------------===//
/// CIRBoolAttr
//===----------------------------------------------------------------------===//

MlirAttribute mlirCirBoolAttrGet(MlirContext ctx, bool value) {
  return wrap(CIRBoolAttr::get(unwrap(ctx), value));
}

bool mlirCirBoolAttrIsA(MlirAttribute attr) {
  return unwrap(attr).isa<CIRBoolAttr>();
}

bool mlirCirBoolAttrValueOf(MlirAttribute attr) {
  return unwrap(attr).cast<CIRBoolAttr>().getValue();
}

//===----------------------------------------------------------------------===//
/// IsizeAttr
//===----------------------------------------------------------------------===//

MlirAttribute mlirCirIsizeAttrGet(MlirContext ctx, uint64_t value) {
  return wrap(IsizeAttr::get(unwrap(ctx), value));
}

bool mlirCirIsizeAttrIsA(MlirAttribute attr) {
  return unwrap(attr).isa<IsizeAttr>();
}

uint64_t mlirCirIsizeAttrValueOf(MlirAttribute attr) {
  auto i = unwrap(attr).cast<IsizeAttr>().getValue();
  return i.getLimitedValue();
}

//===----------------------------------------------------------------------===//
/// CIRFloatAttr
//===----------------------------------------------------------------------===//

MlirAttribute mlirCirFloatAttrGet(MlirContext ctx, double value) {
  return wrap(CIRFloatAttr::get(unwrap(ctx), value));
}

bool mlirCirFloatAttrIsA(MlirAttribute attr) {
  return unwrap(attr).isa<CIRFloatAttr>();
}

double mlirCirFloatAttrValueOf(MlirAttribute attr) {
  return unwrap(attr).cast<CIRFloatAttr>().getValueAsDouble();
}

//===----------------------------------------------------------------------===//
/// AtomAttr
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
/// EndiannessAttr
//===----------------------------------------------------------------------===//

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
