#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/BinaryOpConversions.h"

namespace lumen {
namespace eir {

struct BinaryPushOpConversion : public EIROpConversion<BinaryPushOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      BinaryPushOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    BinaryPushOpOperandAdaptor adaptor(operands);

    Value head = adaptor.head();
    Value tail = adaptor.tail();
    ArrayRef<Value> sizeOpt = adaptor.size();
    Value size;
    if (sizeOpt.size() > 0) {
      size = sizeOpt.front();
    }

    auto pushType = static_cast<uint32_t>(
        op.getAttrOfType<IntegerAttr>("type").getValue().getLimitedValue());

    unsigned unit = 1;
    auto endianness = Endianness::Big;
    bool isSigned = false;
    switch (pushType) {
      case BinarySpecifierType::Bytes:
      case BinarySpecifierType::Bits:
        unit = static_cast<unsigned>(
            op.getAttrOfType<IntegerAttr>("unit").getValue().getLimitedValue());
        break;
      case BinarySpecifierType::Utf8:
      case BinarySpecifierType::Utf16:
      case BinarySpecifierType::Utf32:
        endianness = static_cast<Endianness::Type>(
            op.getAttrOfType<IntegerAttr>("endianness")
                .getValue()
                .getLimitedValue());
        break;
      case BinarySpecifierType::Integer:
        unit = static_cast<unsigned>(
            op.getAttrOfType<IntegerAttr>("unit").getValue().getLimitedValue());
        endianness = static_cast<Endianness::Type>(
            op.getAttrOfType<IntegerAttr>("endianness")
                .getValue()
                .getLimitedValue());
        isSigned = op.getAttrOfType<BoolAttr>("signed").getValue();
        break;
      case BinarySpecifierType::Float:
        unit = static_cast<unsigned>(
            op.getAttrOfType<IntegerAttr>("unit").getValue().getLimitedValue());
        endianness = static_cast<Endianness::Type>(
            op.getAttrOfType<IntegerAttr>("endianness")
                .getValue()
                .getLimitedValue());
        break;
      default:
        llvm_unreachable(
            "invalid binary specifier type encountered during conversion");
    }

    return failure();
  }
};

void populateBinaryOpConversionPatterns(OwningRewritePatternList &patterns,
                                        MLIRContext *context,
                                        LLVMTypeConverter &converter,
                                        TargetInfo &targetInfo) {
  patterns.insert<BinaryPushOpConversion>(context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
