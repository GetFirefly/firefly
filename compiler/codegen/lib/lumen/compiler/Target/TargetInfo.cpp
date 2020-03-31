#include "lumen/compiler/Target/TargetInfo.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Target/TargetMachine.h"

using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::StringRef;
using ::mlir::LLVM::LLVMDialect;
using ::mlir::LLVM::LLVMType;

using ::lumen::Encoding;
using ::lumen::MaskInfo;

using namespace lumen::eir;

namespace lumen {

static APInt make_tag(unsigned pointerSize, uint64_t tag) {
  return APInt(pointerSize, tag, /*signed=*/false);
}

// Used for size calculations
struct AnonymousClosureDefinition {
  uint32_t index;
  char unique[16];
  uint32_t old_unique;
};

// Used for size calculations
union ClosureDefinitionBody {
  unsigned function;
  AnonymousClosureDefinition anon;
};

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
  LLVMType int8Ty = LLVMType::getInt8Ty(&dialect);
  LLVMType int32Ty = LLVMType::getInt32Ty(&dialect);
  LLVMType i8PtrTy = LLVMType::getInt8PtrTy(&dialect);
  LLVMType f64Ty = LLVMType::getDoubleTy(&dialect);
  impl->pointerWidthIntTy = intNTy;
  impl->i1Ty = LLVMType::getInt1Ty(&dialect);
  impl->i8Ty = int8Ty;
  impl->i32Ty = int32Ty;
  impl->opaqueFnTy =
      LLVMType::getFunctionTy(LLVMType::getVoidTy(&dialect), false);

  // BigInt
  impl->bigIntTy = LLVMType::createStructTy(
      &dialect, ArrayRef<LLVMType>(intNTy), StringRef("bigint"));

  // Float
  if (!supportsNanboxing) {
    // Packed floats
    impl->floatTy = LLVMType::createStructTy(
        &dialect, ArrayRef<LLVMType>({intNTy, f64Ty}), StringRef("float"));
  } else {
    // Immediate floats
    impl->floatTy = f64Ty;
  }

  // Cons
  impl->consTy = LLVMType::createStructTy(
      &dialect, ArrayRef<LLVMType>({intNTy, intNTy}), StringRef("cons"));

  // Nil
  auto nilTypeKind = TypeKind::Nil - mlir::Type::FIRST_EIR_TYPE;
  auto nil = lumen_encode_immediate(&impl->encoding, nilTypeKind, 0);
  impl->nil = APInt(pointerSizeInBits, nil, /*signed=*/false);

  // None
  auto noneTypeKind = TypeKind::None - mlir::Type::FIRST_EIR_TYPE;
  auto none = lumen_encode_immediate(&impl->encoding, noneTypeKind, 0);
  impl->none = APInt(pointerSizeInBits, none, /*signed=*/false);

  // Binary types
  ArrayRef<LLVMType> binaryFields({intNTy, intNTy, i8PtrTy});
  impl->binaryTy =
      LLVMType::createStructTy(&dialect, binaryFields, StringRef("binary"));

  // Closure types
  // [i8 x 16]
  impl->uniqueTy = LLVMType::getArrayTy(int8Ty, 16);
  // struct { i8 tag, usize index_or_function_atom, [i8 x 16] unique, i32
  // oldUnique }
  impl->defTy = LLVMType::createStructTy(
      &dialect, ArrayRef<LLVMType>{int8Ty, intNTy, impl->uniqueTy, int32Ty},
      StringRef("closure.definition"));

  // Tags/boxes
  impl->listTag = lumen_list_tag(&impl->encoding);
  impl->listMask = lumen_list_mask(&impl->encoding);
  impl->boxTag = lumen_box_tag(&impl->encoding);
  impl->literalTag = lumen_literal_tag(&impl->encoding);
  impl->immediateMask = lumen_immediate_mask(&impl->encoding);
  impl->headerMask = lumen_header_mask(&impl->encoding);
}

TargetInfo::TargetInfo(const TargetInfo &other)
    : archType(other.archType),
      pointerSizeInBits(other.pointerSizeInBits),
      impl(new TargetInfoImpl(*other.impl)) {}

LLVMType TargetInfo::getConsType() { return impl->consTy; }
LLVMType TargetInfo::getFloatType() { return impl->floatTy; }
LLVMType TargetInfo::getBinaryType() { return impl->binaryTy; }
LLVMType TargetInfo::makeTupleType(LLVMDialect *dialect, unsigned arity) {
  SmallVector<LLVMType, 2> fieldTypes;
  fieldTypes.reserve(1 + arity);
  auto termTy = getUsizeType();
  fieldTypes.push_back(termTy);
  for (auto i = 0; i < arity; i++) {
    fieldTypes.push_back(termTy);
  }
  return LLVMType::createStructTy(dialect, fieldTypes, llvm::None);
}
LLVMType TargetInfo::makeTupleType(LLVMDialect *dialect,
                                   ArrayRef<LLVMType> elementTypes) {
  SmallVector<LLVMType, 3> withHeader;
  withHeader.reserve(1 + elementTypes.size());
  withHeader.push_back(getUsizeType());
  for (auto elemTy : elementTypes) {
    withHeader.push_back(elemTy);
  }
  return LLVMType::createStructTy(dialect, withHeader, llvm::None);
}
LLVMType TargetInfo::getUsizeType() { return impl->pointerWidthIntTy; }
LLVMType TargetInfo::getI1Type() { return impl->i1Ty; }
LLVMType TargetInfo::getI8Type() { return impl->i8Ty; }
LLVMType TargetInfo::getI32Type() { return impl->i32Ty; }
LLVMType TargetInfo::getOpaqueFnType() { return impl->opaqueFnTy; }

/*
#[repr(C)]
pub struct Closure {
    header: Header<Closure>,
    module: Atom,
    definition: Definition,
    arity: u8,
    code: Option<*const ()>,
    env: [Term],
}
*/
LLVMType TargetInfo::makeClosureType(LLVMDialect *dialect, unsigned size) {
  // Construct type of the fields
  auto intNTy = impl->pointerWidthIntTy;
  auto defTy = getClosureDefinitionType();
  auto int8Ty = getI8Type();
  auto voidPtrTy = LLVMType::getFunctionTy(LLVMType::getVoidTy(dialect), false)
                       .getPointerTo();
  auto envTy = LLVMType::getArrayTy(intNTy, size);
  ArrayRef<LLVMType> fields{intNTy, intNTy, defTy, int8Ty, voidPtrTy, envTy};

  // Name the type based on the arity of the env, makes IR more readable
  const char *fmt = "closure%d";
  int bufferSize = std::snprintf(nullptr, 0, fmt, size);
  std::vector<char> buffer(bufferSize + 1);
  std::snprintf(&buffer[0], buffer.size(), fmt, size);
  StringRef typeName(&buffer[0], buffer.size());

  return LLVMType::createStructTy(dialect, fields, typeName.drop_back());
}

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
MaskInfo &TargetInfo::headerMask() const { return impl->headerMask; }

}  // namespace lumen
