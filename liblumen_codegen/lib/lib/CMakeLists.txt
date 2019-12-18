llvm_map_components_to_libnames(Lumen_llvm_libs
  all
)

file(GLOB globbed *.cpp EIR/*.cpp)


set(Lumen_libs
  MLIRAffineOps
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIREDSCInterface
  MLIRExecutionEngine
  MLIRIR
  MLIRLLVMIR
  MLIRLoopOps
  MLIRLoopToStandard
  MLIRParser
  MLIRPass
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRSupport
  MLIRTargetLLVMIR
  MLIRTargetLLVMIRModuleTranslation
  MLIRTransformUtils
  MLIRTransforms
  MLIRTranslateClParser
  MLIRTranslation
  ${Lumen_llvm_libs}
)

add_llvm_library(Lumen
  ${globbed}
  Support/raw_win32_handle_ostream.cpp

  DEPENDS
  OpsIncGen

  LINK_LIBS
  ${Lumen_libs}
)

#target_link_libraries(Lumen PUBLIC EIR)

#target_include_directories(Lumen PUBLIC include/)
#target_include_directories(Lumen PRIVATE ${LLVM_INCLUDE_DIRS})
#target_compile_definitions(Lumen PRIVATE ${LLVM_DEFINITIONS})
#target_compile_options(Lumen PUBLIC -g -fno-rtti)

whole_archive_link(Lumen ${Lumen_libs})

install(TARGETS Lumen
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
