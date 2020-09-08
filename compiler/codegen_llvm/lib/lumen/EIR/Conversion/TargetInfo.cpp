#include "lumen/EIR/Conversion/TargetInfo.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Target/TargetMachine.h"
#include "lumen/EIR/IR/EIRTypes.h"

using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::StringRef;
using ::mlir::MLIRContext;
using ::mlir::LLVM::LLVMType;
using ::mlir::LLVM::LLVMVoidType;

using ::lumen::Encoding;
using ::lumen::MaskInfo;

using namespace lumen::eir;

extern "C" uint32_t lumen_closure_base_size(uint32_t pointerWidth);

namespace lumen {

static APInt make_tag(unsigned pointerSize, uint64_t tag) {
  return APInt(pointerSize, tag, /*signed=*/false);
}

TargetInfo::TargetInfo(llvm::TargetMachine *targetMachine, MLIRContext *ctx)
    : archType(targetMachine->getTargetTriple().getArch()),
      pointerSizeInBits(
          targetMachine->createDataLayout().getPointerSizeInBits(0)),
      impl(new TargetInfoImpl()) {
  // Capture triple as string for use in debug output when necessary
  impl->triple = targetMachine->getTargetTriple().getTriple();
  bool supportsNanboxing = archType == llvm::Triple::ArchType::x86_64;
  impl->encoding = Encoding{pointerSizeInBits, supportsNanboxing};

  // Initialize named types
  LLVMType voidTy = LLVMType::getVoidTy(ctx);
  LLVMType intNTy = LLVMType::getIntNTy(ctx, pointerSizeInBits);
  LLVMType intNPtrTy = intNTy.getPointerTo();
  LLVMType int1Ty = LLVMType::getInt1Ty(ctx);
  LLVMType int8Ty = LLVMType::getInt8Ty(ctx);
  LLVMType int32Ty = LLVMType::getInt32Ty(ctx);
  LLVMType int64Ty = LLVMType::getInt64Ty(ctx);
  LLVMType int8PtrTy = LLVMType::getInt8PtrTy(ctx);
  LLVMType f64Ty = LLVMType::getDoubleTy(ctx);
  LLVMType termTy = intNTy.getPointerTo();
  impl->voidTy = voidTy;
  impl->pointerWidthIntTy = intNTy;
  impl->i1Ty = int1Ty;
  impl->i8Ty = int8Ty;
  impl->i32Ty = int32Ty;
  impl->i64Ty = int64Ty;
  impl->doubleTy = f64Ty;
  impl->opaqueFnTy = LLVMType::getFunctionTy(voidTy, false);

  // BigInt
  impl->bigIntTy = LLVMType::createStructTy(ctx, ArrayRef<LLVMType>(intNTy),
                                            StringRef("bigint"));

  // Float
  if (!supportsNanboxing) {
    // Packed floats
    impl->floatTy = LLVMType::createStructTy(
        ctx, ArrayRef<LLVMType>({intNTy, f64Ty}), StringRef("float"));
  } else {
    // Immediate floats
    impl->floatTy = f64Ty;
  }

  // Cons
  impl->consTy = LLVMType::createStructTy(
      ctx, ArrayRef<LLVMType>({intNTy, intNTy}), StringRef("cons"));

  // Nil
  auto nilTypeKind = TypeKind::Nil;
  auto nil = lumen_encode_immediate(&impl->encoding, nilTypeKind, 0);
  impl->nil = APInt(pointerSizeInBits, nil, /*signed=*/false);

  // None
  auto noneTypeKind = TypeKind::None;
  auto none = lumen_encode_immediate(&impl->encoding, noneTypeKind, 0);
  impl->none = APInt(pointerSizeInBits, none, /*signed=*/false);

  // Binary types
  ArrayRef<LLVMType> binaryFields({intNTy, intNTy, int8PtrTy});
  impl->binaryTy =
      LLVMType::createStructTy(ctx, binaryFields, StringRef("binary"));
  ArrayRef<LLVMType> pushResultFields({intNTy, int1Ty});
  impl->binPushResultTy = LLVMType::createStructTy(ctx, pushResultFields,
                                                   StringRef("binary.pushed"));

  // Match Result
  ArrayRef<LLVMType> matchResultFields({intNTy, intNTy, int1Ty});
  impl->matchResultTy = LLVMType::createStructTy(ctx, matchResultFields,
                                                 StringRef("match.result"));

  // Receives
  impl->recvContextTy = int8Ty.getPointerTo();

  // Closure types
  // [i8 x 16]
  impl->uniqueTy = LLVMType::getArrayTy(int8Ty, 16);
  // struct { u32 tag, usize index_or_function_atom, [i8 x 16] unique, i32
  // oldUnique }
  impl->defTy = LLVMType::createStructTy(
      ctx, ArrayRef<LLVMType>{int32Ty, intNTy, impl->uniqueTy, int32Ty},
      StringRef("closure.definition"));

  // Exception Type (as seen by landing pads)
  impl->exceptionTy =
      LLVMType::createStructTy(ctx, ArrayRef<LLVMType>{int8PtrTy, int32Ty},
                               StringRef("lumen.exception"));

  impl->erlangErrorTy =
    LLVMType::createStructTy(ctx, ArrayRef<LLVMType>{intNTy, intNTy, intNTy, intNPtrTy, int8PtrTy},
                             StringRef("erlang.exception"));

  // Tags/boxes
  impl->listTag = lumen_list_tag(&impl->encoding);
  impl->listMask = lumen_list_mask(&impl->encoding);
  impl->boxTag = lumen_box_tag(&impl->encoding);
  impl->literalTag = lumen_literal_tag(&impl->encoding);
  impl->immediateMask = lumen_immediate_mask(&impl->encoding);
  impl->headerMask = lumen_header_mask(&impl->encoding);

  auto maxAllowedImmediateVal =
      APInt(64, impl->immediateMask.maxAllowedValue, /*signed=*/false);
  impl->immediateBits = maxAllowedImmediateVal.getActiveBits();
}

TargetInfo::TargetInfo(const TargetInfo &other)
    : archType(other.archType),
      pointerSizeInBits(other.pointerSizeInBits),
      impl(new TargetInfoImpl(*other.impl)) {}

LLVMType TargetInfo::makeTupleType(unsigned arity) {
  auto termTy = getUsizeType();
  if (arity == 0) {
    return LLVMType::getStructTy(termTy.getContext(),
                                 ArrayRef<LLVMType>{termTy});
  }
  SmallVector<LLVMType, 2> fieldTypes;
  fieldTypes.reserve(1 + arity);
  fieldTypes.push_back(termTy);
  for (auto i = 0; i < arity; i++) {
    fieldTypes.push_back(termTy);
  }
  return LLVMType::getStructTy(termTy.getContext(), fieldTypes);
}
LLVMType TargetInfo::makeTupleType(ArrayRef<LLVMType> elementTypes) {
  auto termTy = getUsizeType();
  if (elementTypes.size() == 0) {
    return LLVMType::getStructTy(termTy.getContext(),
                                 ArrayRef<LLVMType>{termTy});
  }
  SmallVector<LLVMType, 3> withHeader;
  withHeader.reserve(1 + elementTypes.size());
  withHeader.push_back(termTy);
  for (auto elemTy : elementTypes) {
    withHeader.push_back(elemTy);
  }
  return LLVMType::getStructTy(termTy.getContext(), withHeader);
}

/*
#[repr(C)]
pub struct Closure {
    header: Header<Closure>,
    module: Atom,
    arity: u32,
    definition: Definition,
    code: Option<NonNull<*const c_void>>,
    env: [Term],
}
*/
LLVMType TargetInfo::makeClosureType(unsigned size) {
  // Construct type of the fields
  auto intNTy = impl->pointerWidthIntTy;
  auto defTy = getClosureDefinitionType();
  auto int32Ty = getI32Type();
  auto voidTy = impl->voidTy;
  auto voidFnPtrTy =
      LLVMType::getFunctionTy(voidTy, false).getPointerTo();
  auto envTy = LLVMType::getArrayTy(intNTy, size);
  ArrayRef<LLVMType> fields{intNTy, intNTy, int32Ty, defTy, voidFnPtrTy, envTy};

  // Name the type based on the arity of the env, makes IR more readable
  const char *fmt = "closure%d";
  int bufferSize = std::snprintf(nullptr, 0, fmt, size);
  std::vector<char> buffer(bufferSize + 1);
  std::snprintf(&buffer[0], buffer.size(), fmt, size);
  StringRef typeName(&buffer[0], buffer.size());

  return LLVMType::createStructTy(intNTy.getContext(), fields, typeName.drop_back());
}

APInt TargetInfo::encodeImmediate(unsigned type, uint64_t value) {
  auto encoded = lumen_encode_immediate(&impl->encoding,
                                        static_cast<uint32_t>(type), value);
  return APInt(pointerSizeInBits, encoded, /*signed=*/false);
}

APInt TargetInfo::encodeHeader(unsigned type, uint64_t arity) {
  auto encoded =
      lumen_encode_header(&impl->encoding, static_cast<uint32_t>(type), arity);
  return APInt(pointerSizeInBits, encoded, /*signed=*/false);
}

APInt &TargetInfo::getNilValue() const { return impl->nil; }
APInt &TargetInfo::getNoneValue() const { return impl->none; }

uint64_t TargetInfo::listTag() const { return impl->listTag; }
uint64_t TargetInfo::listMask() const { return impl->listMask; }
uint64_t TargetInfo::boxTag() const { return impl->boxTag; }
uint64_t TargetInfo::literalTag() const { return impl->literalTag; }
uint32_t TargetInfo::closureHeaderArity(uint32_t envLen) const {
  uint32_t wordSize;
  if (pointerSizeInBits == 64) {
    wordSize = 8;
  } else {
    assert(pointerSizeInBits == 32 && "unsupported pointer width");
    wordSize = 4;
  }
  auto totalBytes =
      lumen_closure_base_size(pointerSizeInBits) + (envLen * wordSize);
  return (totalBytes / wordSize) - 1;
}
MaskInfo &TargetInfo::immediateMask() const { return impl->immediateMask; }
MaskInfo &TargetInfo::headerMask() const { return impl->headerMask; }

}  // namespace lumen
