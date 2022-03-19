#ifndef EIR_TARGET_PLATFORM_BUILDER_H
#define EIR_TARGET_PLATFORM_BUILDER_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "lumen/EIR/IR/TargetPlatform.h"

namespace mlir {
class ModuleOp;
class Type;
class IntegerType;
class IntegerAttr;
class Value;
class Location;
class Operation;
}  // namespace mlir

namespace lumen {
namespace eir {

/// TargetPlatformBuilder extends OpBuilder with functionality for constructing
/// terms that are correctly encoded for the provided target platform.
class TargetPlatformBuilder : public ::mlir::OpBuilder {
   public:
    explicit TargetPlatformBuilder(const ::mlir::OpBuilder &otherBuilder,
                                   const TargetPlatform &platform)
        : ::mlir::OpBuilder(otherBuilder), platform(platform) {}

    explicit TargetPlatformBuilder(const TargetPlatformBuilder &other)
        : ::mlir::OpBuilder(other), platform(other.platform) {}

   protected:
    TargetPlatform platform;

   public:
    bool useNanboxedFloats() const {
        return platform.getEncoder().supportsNanboxing();
    }

    inline ::mlir::IntegerType getUsizeType() {
        return getIntegerType(platform.getEncoder().getPointerWidth(),
                              /*signed=*/false);
    }
    inline ::mlir::IntegerType getI8Type() {
        return getIntegerType(8, /*signed=*/true);
    }
    inline ::mlir::IntegerType getI16Type() {
        return getIntegerType(16, /*signed=*/true);
    }

    ::mlir::Type getTokenType();
    ::mlir::Type getVoidType();
    ::mlir::Type getOpaqueTermType();
    ::mlir::Type getOpaqueTermTypeAddr0();
    ::mlir::Type getPointerType(::mlir::Type pointeeType,
                                unsigned addrspace = 0);
    ::mlir::Type getConsType();
    ::mlir::Type getErlangTupleType(unsigned arity);
    ::mlir::Type getFloatType();
    ::mlir::Type getBigIntType();
    ::mlir::Type getBinaryType();
    ::mlir::Type getBinaryBuilderType();
    ::mlir::Type getBinaryPushResultType();
    ::mlir::Type getMatchResultType();
    ::mlir::Type getTraceRefType();
    ::mlir::Type getRecvContextType();
    ::mlir::Type getClosureType(unsigned size);
    ::mlir::Type getClosureDefinitionType();
    ::mlir::Type getExceptionType();
    ::mlir::Type getErrorType();

    ::mlir::IntegerAttr getU32Attr(int32_t i);

    ::mlir::Operation *getOrInsertFunction(
        ::mlir::ModuleOp mod, ::llvm::StringRef symbol, ::mlir::Type resultTy,
        ::llvm::ArrayRef<::mlir::Type> argTypes,
        ::llvm::ArrayRef<::mlir::NamedAttribute> attrs = {});

    ::mlir::Value buildMalloc(::mlir::ModuleOp mod, ::mlir::Type ty,
                              unsigned allocTy, ::mlir::Value arity);

    ::mlir::Value encodeList(::mlir::Value cons, bool isLiteral = false);
    ::mlir::Value encodeBox(::mlir::Value val);
    ::mlir::Value encodeLiteral(::mlir::Value val);
    ::mlir::Value encodeImmediate(::mlir::ModuleOp mod, ::mlir::Location loc,
                                  ::mlir::Type ty, ::mlir::Value val);
    ::llvm::APInt encodeImmediateConstant(uint32_t type, uint64_t value) {
        return platform.getEncoder().encodeImmediate(type, value);
    }
    ::llvm::APInt encodeHeaderConstant(uint32_t type, uint64_t arity) {
        return platform.getEncoder().encodeHeader(type, arity);
    }
    ::mlir::Value decodeBox(::mlir::Type innerTy, ::mlir::Value box);
    ::mlir::Value decodeList(::mlir::Value box);
    ::mlir::Value decodeImmediate(::mlir::Value val);
};

}  // namespace eir
}  // namespace lumen

#endif
