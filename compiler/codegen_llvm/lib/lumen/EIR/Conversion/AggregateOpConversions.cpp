#include "lumen/EIR/Conversion/AggregateOpConversions.h"

namespace lumen {
namespace eir {

struct ConsOpConversion : public EIROpConversion<ConsOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      ConsOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    ConsOpAdaptor adaptor(operands);

    auto termTy = ctx.getUsizeType();
    auto termPtrTy = termTy.getPointerTo();
    auto i32Ty = ctx.getI32Type();
    auto consTy = ctx.targetInfo.getConsType();

    auto head = adaptor.head();
    auto tail = adaptor.tail();

    // Allocate header on heap, write values to header, then box
    OpaqueTermType kind = rewriter.getType<ConsType>();
    Value zero = llvm_constant(i32Ty, ctx.getI32Attr(0));
    Value one = llvm_constant(i32Ty, ctx.getI32Attr(1));
    Value arity = llvm_zext(termTy, zero);
    // TODO(pauls): We should optimize this for allocating multiple
    // cells by providing an optional pointer and index at which to
    // allocate this cell, by offsetting the pointer by `index * sizeof(cell)`
    // and then storing directly into that memory
    Value cellPtr = ctx.buildMalloc(consTy, TypeKind::Cons, arity);
    ArrayRef<Value> headIndices{zero, zero};
    Value headPtr = llvm_gep(termPtrTy, cellPtr, headIndices);
    llvm_store(head, headPtr);
    ArrayRef<Value> tailIndices{zero, one};
    Value tailPtr = llvm_gep(termPtrTy, cellPtr, tailIndices);
    llvm_store(tail, tailPtr);

    auto boxed = ctx.encodeList(cellPtr);
    rewriter.replaceOp(op, boxed);
    return success();
  }
};

struct ListOpConversion : public EIROpConversion<ListOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      ListOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    ListOpAdaptor adaptor(operands);

    auto elements = adaptor.elements();
    auto numElements = elements.size();

    if (numElements == 0) {
      Value nil = eir_nil();
      rewriter.replaceOp(op, nil);
      return success();
    }

    // Lower to single cons cell if it fits
    if (numElements < 2) {
      Value head = elements.front();
      Value list = eir_cons(head, eir_nil());
      rewriter.replaceOp(op, list);
      return success();
    }

    unsigned cellsRequired = numElements;
    unsigned currentIndex = numElements;

    Value list;
    while (currentIndex > 0) {
      if (!list) {
        Value tail = elements[--currentIndex];
        Value head = elements[--currentIndex];
        list = eir_cons(head, tail);
      } else {
        Value head = elements[--currentIndex];
        list = eir_cons(head, list);
      }
    }

    rewriter.replaceOp(op, list);
    return success();
  }
};

struct TupleOpConversion : public EIROpConversion<TupleOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      TupleOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    TupleOpAdaptor adaptor(operands);

    auto termTy = ctx.getUsizeType();
    auto termPtrTy = termTy.getPointerTo();
    auto i32Ty = ctx.getI32Type();
    auto elements = adaptor.elements();
    auto numElements = elements.size();
    auto tupleTy = ctx.getTupleType(numElements);

    // Allocate header on heap, write values to header, then box
    Value arity = llvm_constant(termTy, ctx.getIntegerAttr(numElements));
    Value ptr = ctx.buildMalloc(tupleTy, TypeKind::Tuple, arity);

    Value zero = llvm_constant(i32Ty, ctx.getI32Attr(0));
    auto headerRaw = ctx.targetInfo.encodeHeader(TypeKind::Tuple, numElements);
    ArrayRef<Value> headerTermIndices{zero, zero};
    Value headerTermPtr = llvm_gep(termPtrTy, ptr, headerTermIndices);
    llvm_store(llvm_constant(termTy, ctx.getIntegerAttr(headerRaw)),
               headerTermPtr);

    for (auto i = 0; i < numElements; i++) {
      auto element = elements[i];
      auto elementTy = tupleTy.getStructElementType(i + 1).getPointerTo();
      auto idx = llvm_constant(i32Ty, ctx.getI32Attr(i + 1));
      ArrayRef<Value> elementIndices{zero, idx};
      auto elementPtr = llvm_gep(elementTy, ptr, elementIndices);
      llvm_store(element, elementPtr);
    }

    // Box the allocated tuple
    auto boxed = ctx.encodeBox(ptr);
    rewriter.replaceOp(op, boxed);
    return success();
  }
};

void populateAggregateOpConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *context,
                                           EirTypeConverter &converter,
                                           TargetInfo &targetInfo) {
  patterns.insert<ConsOpConversion, ListOpConversion,
                  TupleOpConversion>(context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
