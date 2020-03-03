#include "lumen/compiler/Target/TargetInfo.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Target/TargetMachine.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::StringRef;
using ::mlir::LLVM::LLVMDialect;
using ::mlir::LLVM::LLVMType;

using ::lumen::Encoding;
using ::lumen::MaskInfo;

using namespace lumen::eir;

extern "C" bool lumen_is_type(Encoding *encoding, uint32_t type,
                              uint64_t value);
extern "C" uint64_t lumen_encode_immediate(Encoding *encoding, uint32_t type,
                                           uint64_t value);
extern "C" uint64_t lumen_encode_header(Encoding *encoding, uint32_t type,
                                        uint64_t arity);
extern "C" uint64_t lumen_list_tag(Encoding *encoding);
extern "C" uint64_t lumen_box_tag(Encoding *encoding);
extern "C" uint64_t lumen_literal_tag(Encoding *encoding);
extern "C" MaskInfo lumen_immediate_mask(Encoding *encoding);
extern "C" uint64_t lumen_list_mask(Encoding *encoding);

namespace lumen {

static APInt make_tag(unsigned pointerSize, uint64_t tag) {
  return APInt(pointerSize, tag, /*signed=*/false);
}

TargetInfo::TargetInfo(llvm::TargetMachine *targetMachine, LLVMDialect &dialect)
    : archType(targetMachine->getTargetTriple().getArch()),
      pointerSizeInBits(
          targetMachine->createDataLayout().getPointerSizeInBits(0)),
      impl(new TargetInfoImpl()) {
  // Capture triple as string for use in debug output when necessary
  impl->triple = targetMachine->getTargetTriple().getTriple();
  bool supportsNanboxing = archType == llvm::Triple::ArchType::x86_64;
  impl->encoding = Encoding{pointerSizeInBits, supportsNanboxing};

  // Initialize named types
  LLVMType intNTy = LLVMType::getIntNTy(&dialect, pointerSizeInBits);
  LLVMType i8PtrTy = LLVMType::getInt8PtrTy(&dialect);
  LLVMType f64Ty = LLVMType::getDoubleTy(&dialect);
  impl->pointerWidthIntTy = intNTy;
  impl->i1Ty = LLVMType::getInt1Ty(&dialect);
  // auto termTy = LLVMType::createStructTy(&dialect,
  // ArrayRef<LLVMType>(intNTy), StringRef("term"));
  auto termTy = intNTy;
  impl->termTy = termTy;
  impl->atomTy = LLVMType::createStructTy(&dialect, ArrayRef<LLVMType>(intNTy),
                                          StringRef("atom"));
  impl->nilTy = LLVMType::createStructTy(&dialect, ArrayRef<LLVMType>(intNTy),
                                         StringRef("nil"));
  impl->noneTy = LLVMType::createStructTy(&dialect, ArrayRef<LLVMType>(intNTy),
                                          StringRef("none"));
  impl->fixnumTy = LLVMType::createStructTy(
      &dialect, ArrayRef<LLVMType>(intNTy), StringRef("fixnum"));
  impl->bigIntTy = LLVMType::createStructTy(
      &dialect, ArrayRef<LLVMType>(intNTy), StringRef("bigint"));
  if (!supportsNanboxing) {
    // Packed floats
    impl->floatTy = LLVMType::createStructTy(
        &dialect, ArrayRef<LLVMType>({intNTy, f64Ty}), StringRef("float"));
  } else {
    // Immediate floats
    impl->floatTy = f64Ty;
  }
  impl->consTy = LLVMType::createStructTy(
      &dialect, ArrayRef<LLVMType>({termTy, termTy}), StringRef("cons"));
  auto nilTypeKind = TypeKind::Nil - mlir::Type::FIRST_EIR_TYPE;
  auto nil = lumen_encode_immediate(&impl->encoding, nilTypeKind, 0);
  impl->nil = APInt(pointerSizeInBits, nil, /*signed=*/false);
  auto noneTypeKind = TypeKind::None - mlir::Type::FIRST_EIR_TYPE;
  auto none = lumen_encode_immediate(&impl->encoding, noneTypeKind, 0);
  impl->none = APInt(pointerSizeInBits, none, /*signed=*/false);

  ArrayRef<LLVMType> binaryFields({intNTy, intNTy, i8PtrTy});
  impl->binaryTy =
      LLVMType::createStructTy(&dialect, binaryFields, StringRef("binary"));

  impl->listTag = lumen_list_tag(&impl->encoding);
  impl->boxTag = lumen_box_tag(&impl->encoding);
  impl->literalTag = lumen_literal_tag(&impl->encoding);
  impl->immediateMask = lumen_immediate_mask(&impl->encoding);
  impl->listMask = lumen_list_mask(&impl->encoding);
}

TargetInfo::TargetInfo(const TargetInfo &other)
    : archType(other.archType),
      pointerSizeInBits(other.pointerSizeInBits),
      impl(new TargetInfoImpl(*other.impl)) {}

LLVMType TargetInfo::getHeaderType() { return impl->pointerWidthIntTy; }
LLVMType TargetInfo::getTermType() { return impl->termTy; }
LLVMType TargetInfo::getConsType() { return impl->consTy; }
LLVMType TargetInfo::getFloatType() { return impl->floatTy; }
LLVMType TargetInfo::getFixnumType() { return impl->fixnumTy; }
LLVMType TargetInfo::getAtomType() { return impl->atomTy; }
LLVMType TargetInfo::getBinaryType() { return impl->binaryTy; }
LLVMType TargetInfo::getNilType() { return impl->nilTy; }
LLVMType TargetInfo::getNoneType() { return impl->noneTy; }
LLVMType TargetInfo::makeTupleType(LLVMDialect *dialect,
                                   ArrayRef<LLVMType> elementTypes) {
  return LLVMType::createStructTy(dialect, elementTypes, llvm::None);
}
LLVMType TargetInfo::getI1Type() { return impl->i1Ty; }

APInt TargetInfo::encodeImmediate(unsigned type, uint64_t value) {
  auto t = type - mlir::Type::FIRST_EIR_TYPE;
  auto encoded =
      lumen_encode_immediate(&impl->encoding, static_cast<uint32_t>(t), value);
  return APInt(pointerSizeInBits, encoded, /*signed=*/false);
}

APInt TargetInfo::encodeHeader(unsigned type, uint64_t arity) {
  auto t = type - mlir::Type::FIRST_EIR_TYPE;
  auto encoded =
      lumen_encode_header(&impl->encoding, static_cast<uint32_t>(t), arity);
  return APInt(pointerSizeInBits, encoded, /*signed=*/false);
}

APInt &TargetInfo::getNilValue() const { return impl->nil; }
APInt &TargetInfo::getNoneValue() const { return impl->none; }

uint64_t TargetInfo::listTag() const { return impl->listTag; }
uint64_t TargetInfo::listMask() const { return impl->listMask; }
uint64_t TargetInfo::boxTag() const { return impl->boxTag; }
uint64_t TargetInfo::literalTag() const { return impl->literalTag; }
MaskInfo &TargetInfo::immediateMask() const { return impl->immediateMask; }

}  // namespace lumen
