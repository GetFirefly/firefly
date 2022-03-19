#pragma once

#include <stdint.h>

namespace lumen {

extern "C" struct Encoding {
  uint32_t pointerWidth;
  bool supportsNanboxing;
};

extern "C" struct MaskInfo {
  int32_t shift;
  uint64_t mask;
  uint64_t maxAllowedValue;

  bool requiresShift() const { return shift != 0; }
};

namespace TermKind {
enum Kind {
  None = 0,
  Term = 1,
  List = 2,
  Number = 3,
  Integer = 4,
  Float = 5,
  Atom = 6,
  Bool = 7,
  Isize = 8,
  BigInt = 9,
  Nil = 10,
  Cons = 11,
  Tuple = 12,
  Map = 13,
  Fun = 14,
  Bits = 15,
  Heapbin = 16,
  Procbin = 17,
  Box = 18,
  Pid = 19,
  Port = 20,
  Reference = 21,
};
}

} // namespace lumen

extern "C" bool lumen_is_type(const ::lumen::Encoding *encoding,
                              ::lumen::TermKind::Kind type, uint64_t value);
extern "C" uint64_t lumen_encode_immediate(const ::lumen::Encoding *encoding,
                                           ::lumen::TermKind::Kind type,
                                           uint64_t value);
extern "C" uint64_t lumen_encode_header(const ::lumen::Encoding *encoding,
                                        ::lumen::TermKind::Kind type,
                                        uint64_t arity);
extern "C" uint64_t lumen_immediate_tag(const ::lumen::Encoding *encoding,
                                        ::lumen::TermKind::Kind type);
extern "C" uint64_t lumen_header_tag(const ::lumen::Encoding *encoding,
                                     ::lumen::TermKind::Kind type);
extern "C" uint64_t lumen_list_tag(const ::lumen::Encoding *encoding);
extern "C" uint64_t lumen_list_mask(const ::lumen::Encoding *encoding);
extern "C" uint64_t lumen_box_tag(const ::lumen::Encoding *encoding);
extern "C" uint64_t lumen_literal_tag(const ::lumen::Encoding *encoding);
extern "C" ::lumen::MaskInfo
lumen_immediate_mask(const ::lumen::Encoding *encoding);
extern "C" ::lumen::MaskInfo
lumen_header_mask(const ::lumen::Encoding *encoding);
