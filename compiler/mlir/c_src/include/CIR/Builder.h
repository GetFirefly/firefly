#pragma once

#include "CIR-c/Builder.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace cir {
struct InsertPoint {
  explicit InsertPoint(mlir::Block *block, mlir::Block::iterator point)
      : block(block), point(point) {}

  mlir::Block *block;
  mlir::Block::iterator point;
};
} // namespace cir
} // namespace mlir

DEFINE_C_API_PTR_METHODS(MlirBuilder, mlir::Builder);
DEFINE_C_API_PTR_METHODS(MlirOpBuilder, mlir::OpBuilder);
DEFINE_C_API_PTR_METHODS(MlirInsertPoint, mlir::cir::InsertPoint);
