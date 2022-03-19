#ifndef EIR_TARGETPLATFORM_H
#define EIR_TARGETPLATFORM_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "mlir/IR/Builders.h"

#include "lumen/EIR/IR/EIRAttributes.h"

//===----------------------------------------------------------------------===//
// Structs
//===----------------------------------------------------------------------===//

#include "lumen/term/Encoding.h"

#include <memory>
#include <stdbool.h>
#include <stdint.h>

namespace lumen {
namespace eir {

namespace detail {
struct TargetPlatformEncoderImpl {
    explicit TargetPlatformEncoderImpl(const ::llvm::Triple &triple);

    Encoding encoding;

    unsigned pointerWidth, immediateWidth = 0;

    uint64_t listTag, listMask, boxTag, literalTag = 0;

    uint32_t nilVal, noneVal = 0;

    ::lumen::MaskInfo immediateMask, headerMask;
};
}  // namespace detail

/// TargetPlatformEncoder is a trivial class which can be used to perform
/// low-level encoding of terms.
///
/// It is a wrapper around a shared reference to the actual encoding metadata,
/// so it may be copied/moved freely.
class TargetPlatformEncoder {
   public:
    explicit TargetPlatformEncoder(const ::llvm::Triple &triple)
        : impl(std::make_shared<detail::TargetPlatformEncoderImpl>(triple)) {}

    TargetPlatformEncoder(TargetPlatformEncoder &other) : impl(other.impl) {}
    TargetPlatformEncoder(TargetPlatformEncoder &&other)
        : impl(std::move(other.impl)) {}

   private:
    std::shared_ptr<detail::TargetPlatformEncoderImpl> impl;

   public:
    inline unsigned getPointerWidth() const { return impl->pointerWidth; }
    inline unsigned getImmediateWidth() const { return impl->immediateWidth; }

    inline uint64_t getListTag() const { return impl->listTag; }
    inline uint64_t getListMask() const { return impl->listMask; }
    inline uint64_t getBoxTag() const { return impl->boxTag; }
    inline uint64_t getLiteralTag() const { return impl->literalTag; }
    inline uint32_t getClosureHeaderArity(uint32_t envLen) const;

    inline ::lumen::MaskInfo &getImmediateMask() const {
        return impl->immediateMask;
    }
    inline ::lumen::MaskInfo &getHeaderMask() const { return impl->headerMask; }

    ::llvm::APInt getNilValue() const;
    ::llvm::APInt getNoneValue() const;

    ::llvm::APInt encodeImmediate(uint32_t type, uint64_t value);
    ::llvm::APInt encodeHeader(uint32_t type, uint64_t arity);

    inline bool supportsNanboxing() const {
        return impl->encoding.supportsNanboxing;
    }

    bool isValidImmediateValue(::llvm::APInt &value) const {
        return value.getMinSignedBits() <= impl->immediateWidth;
    }

    bool isValidHeaderValue(::llvm::APInt &value) const {
        if (!isValidImmediateValue(value)) return false;
        auto headerMask = getHeaderMask();
        return value.getLimitedValue() <= headerMask.maxAllowedValue;
    }
};

namespace detail {
struct TargetPlatformImpl {
    explicit TargetPlatformImpl(const ::llvm::Triple &triple)
        : triple(triple), encoder(triple) {
        pointerWidth = encoder.getPointerWidth();
    }

    unsigned pointerWidth;

    ::llvm::Triple triple;

    TargetPlatformEncoder encoder;
};
}  // namespace detail

/// TargetPlatform provides information about the current target
/// platform/architecture, including the size in bits of pointers, the maximum
/// bit width of immediates, what features are supported on the target, and
/// more.
///
/// This class is actually a wrapper around a shared reference, so it may be
/// copied/moved freely.
class TargetPlatform {
   public:
    explicit TargetPlatform(::llvm::StringRef triple)
        : TargetPlatform(::llvm::Triple(triple)) {}

    explicit TargetPlatform(const ::llvm::Triple &triple)
        : impl(std::make_shared<detail::TargetPlatformImpl>(triple)) {}

    TargetPlatform(const TargetPlatform &other) : impl(other.impl) {}
    TargetPlatform(TargetPlatform &&other) : impl(std::move(other.impl)) {}

   private:
    std::shared_ptr<detail::TargetPlatformImpl> impl = nullptr;

   public:
    inline unsigned getPointerWidth() const { return impl->pointerWidth; }

    inline ::llvm::Triple &getTriple() const { return impl->triple; }

    inline TargetPlatformEncoder &getEncoder() const { return impl->encoder; }
};

}  // namespace eir
}  // namespace lumen

#endif  // EIR_TARGETPLATFORM_H
