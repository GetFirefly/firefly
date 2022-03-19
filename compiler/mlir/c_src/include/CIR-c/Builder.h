#pragma once

#include "mlir-c/Support.h"

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

#ifdef __cplusplus
extern "C" {
#endif

DEFINE_C_API_STRUCT(MlirBuilder, void);
DEFINE_C_API_STRUCT(MlirOpBuilder, void);
DEFINE_C_API_STRUCT(MlirInsertPoint, void);

#ifdef __cplusplus
} // extern "C"
#endif

#undef DEFINE_C_API_STRUCT
