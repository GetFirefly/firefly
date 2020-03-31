#ifndef LUMEN_TERM_ENCODING_H
#define LUMEN_TERM_ENCODING_H

#include <stdint.h>

namespace lumen {

extern "C" struct Encoding {
  uint32_t pointerWidth;
  bool supportsNanboxing;
};

extern "C" struct MaskInfo {
  int32_t shift;
  uint64_t mask;

  bool requiresShift() const { return shift != 0; }
};

}// namespace lumen

extern "C" bool lumen_is_type(::lumen::Encoding *encoding, uint32_t type,
                              uint64_t value);
extern "C" uint64_t lumen_encode_immediate(::lumen::Encoding *encoding, uint32_t type,
                                           uint64_t value);
extern "C" uint64_t lumen_encode_header(::lumen::Encoding *encoding, uint32_t type,
                                        uint64_t arity);
extern "C" uint64_t lumen_list_tag(::lumen::Encoding *encoding);
extern "C" uint64_t lumen_list_mask(::lumen::Encoding *encoding);
extern "C" uint64_t lumen_box_tag(::lumen::Encoding *encoding);
extern "C" uint64_t lumen_literal_tag(::lumen::Encoding *encoding);
extern "C" ::lumen::MaskInfo lumen_immediate_mask(::lumen::Encoding *encoding);
extern "C" ::lumen::MaskInfo lumen_header_mask(::lumen::Encoding *encoding);

#endif
