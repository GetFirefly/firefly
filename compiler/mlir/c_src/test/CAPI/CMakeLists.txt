# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own standalone dialect.
add_mlir_aggregate(CIRCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  MLIRCAPIRegistration
  CIRCAPI
)

add_llvm_executable(cir-capi-test
  cir-capi-test.c
)
llvm_update_compile_flags(cir-capi-test)
target_link_libraries(cir-capi-test
  PRIVATE CIRCAPITestLib)
