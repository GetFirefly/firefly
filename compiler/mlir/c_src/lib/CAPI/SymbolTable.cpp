#include "mlir/IR/SymbolTable.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Attributes.h"

#include "CIR-c/SymbolTable.h"

using ::mlir::StringAttr;

static inline mlir::SymbolTable::Visibility unwrap(MlirVisibility v) {
  switch (v) {
  case MlirVisibilityPublic:
    return mlir::SymbolTable::Visibility::Public;
  case MlirVisibilityPrivate:
    return mlir::SymbolTable::Visibility::Private;
  case MlirVisibilityNested:
    return mlir::SymbolTable::Visibility::Nested;
  }
}

static inline MlirVisibility wrap(mlir::SymbolTable::Visibility v) {
  switch (v) {
  case mlir::SymbolTable::Visibility::Public:
    return MlirVisibilityPublic;
  case mlir::SymbolTable::Visibility::Nested:
    return MlirVisibilityNested;
  case mlir::SymbolTable::Visibility::Private:
    return MlirVisibilityPrivate;
  }
}

MlirOperation mlirSymbolTableGetNearestSymbolTable(MlirOperation from) {
  return wrap(mlir::SymbolTable::getNearestSymbolTable(unwrap(from)));
}

MlirOperation mlirSymbolTableLookupNearestSymbolFrom(MlirOperation from,
                                                     MlirStringRef symbol) {
  auto op = unwrap(from);
  auto attr = StringAttr::get(op->getContext(), unwrap(symbol));
  return wrap(mlir::SymbolTable::lookupNearestSymbolFrom(op, attr));
}

MlirOperation mlirSymbolTableLookupIn(MlirOperation in, MlirStringRef symbol) {
  auto op = unwrap(in);
  auto attr = StringAttr::get(op->getContext(), unwrap(symbol));
  return wrap(mlir::SymbolTable::lookupSymbolIn(op, attr));
}

int mlirSymbolTableSymbolKnownUseEmpty(MlirStringRef symbol,
                                       MlirOperation from) {
  auto op = unwrap(from);
  auto attr = StringAttr::get(op->getContext(), unwrap(symbol));
  return mlir::SymbolTable::symbolKnownUseEmpty(attr, op);
}

MlirVisibility mlirSymbolTableGetSymbolVisibility(MlirOperation symbol) {
  return wrap(mlir::SymbolTable::getSymbolVisibility(unwrap(symbol)));
}

void mlirSymbolTableSetSymbolVisibility(MlirOperation symbol,
                                        MlirVisibility visibility) {
  mlir::SymbolTable::setSymbolVisibility(unwrap(symbol), unwrap(visibility));
}

MlirStringRef mlirSymbolTableGetSymbolName(MlirOperation symbol) {
  StringAttr attr = mlir::SymbolTable::getSymbolName(unwrap(symbol));
  return wrap(attr.strref());
}

void mlirSymbolTableSetSymbolName(MlirOperation symbol, MlirStringRef name) {
  mlir::SymbolTable::setSymbolName(unwrap(symbol), unwrap(name));
}

MlirStringRef mlirSymbolTableGetSymbolAttrName() {
  return wrap(mlir::SymbolTable::getSymbolAttrName());
}

MlirStringRef mlirSymbolTableGetVisibilityAttrName() {
  return wrap(mlir::SymbolTable::getVisibilityAttrName());
}
