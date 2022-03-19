#include "lumen/EIR/IR/TargetPlatform.h"

using namespace ::lumen;
using namespace ::lumen::eir;

using ::llvm::APInt;
using ::llvm::StringRef;
using ::llvm::Triple;

extern "C" uint32_t lumen_closure_size(uint32_t pointerWidth, uint32_t envLen);

/// TargetPlatformEncoder

TargetPlatformEncoderImpl::TargetPlatformEncoderImpl(const Triple &triple) {
    auto archType = triple.getArch();

    if (triple.isArch64Bit())
        pointerWidth = 64;
    else if (triple.isArch32Bit())
        pointerWidth = 32;
    else if (triple.isArch16Bit())
        pointerWidth = 16;
    else
        pointerWidth = 0;

    bool supportsNanboxing = archType == Triple::x86_64;

    encoding = Encoding{pointerWidth, supportsNanboxing};

    // Tags/boxes
    listTag = lumen_list_tag(&encoding);
    listMask = lumen_list_mask(&encoding);
    boxTag = lumen_box_tag(&encoding);
    literalTag = lumen_literal_tag(&encoding);
    immediateMask = lumen_immediate_mask(&encoding);
    headerMask = lumen_header_mask(&encoding);

    auto maxAllowedImmediateVal =
        APInt(pointerWidth == 0 ? 64 : pointerWidth,
              immediateMask.maxAllowedValue, /*signed=*/false);
    immediateWidth = maxAllowedImmediateVal.getActiveBits();

    // Nil
    nilVal = lumen_encode_immediate(&encoding, TypeKind::Nil, 0);

    // None
    noneVal = lumen_encode_immediate(&encoding, TypeKind::None, 0);
}

APInt TargetPlatformEncoder::getNilValue() const {
    return APInt(impl->pointerWidth, impl->nilValue, /*signed=*/false);
}

APInt TargetPlatformEncoder::getNoneValue() const {
    return APInt(impl->pointerWidth, impl->noneValue, /*signed=*/false);
}

uint32_t TargetPlatformEncoder::closureHeaderArity(uint32_t envLen) const {
    uint32_t pointerWidth, wordSize;

    pointerWidth = impl->pointerWidth;
    if (pointerWidth == 64) {
        wordSize = 8;
    } else {
        assert(pointerWidth == 32 && "unsupported pointer width");
        wordSize = 4;
    }
    // This is the full size of the closure in memory (in bytes)
    auto totalBytes = lumen_closure_size(pointerWidth, envLen);
    // Converted to words (pointer-size integer in bytes)
    auto words = totalBytes / wordSize;
    // Header arity is in words, and does _not_ include the header word
    return words - 1;
}

::llvm::APInt TargetPlatformEncoder::encodeImmediate(uint32_t type,
                                                     uint64_t value) const {
    auto encoded = lumen_encode_immediate(&impl->encoding, type, value);
    return APInt(impl->pointerWidth, encoded, /*signed=*/false);
}

::llvm::APInt TargetPlatformEncoder::encodeHeader(uint32_t type,
                                                  uint64_t arity) const {
    auto encoded = lumen_encode_header(&impl->encoding, type, arity);
    return APInt(impl->pointerWidth, encoded, /*signed=*/false);
}
