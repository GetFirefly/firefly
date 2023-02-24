#pragma once

#include "mlir-c/Support.h"
#include "mlir/CAPI/Wrap.h"

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name
