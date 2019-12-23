llvm_map_components_to_libnames(Lumen_llvm_libs
  all
)

file(GLOB globbed *.cpp Support/*.cpp EIR/*.cpp)

set(Lumen_mlir_libs
  MLIRAnalysis
  MLIRAffineOps
  MLIRAffineToStandard
  MLIRDialect
  MLIREDSC
  MLIREDSCInterface
  MLIRExecutionEngine
  MLIRFxpMathOps
  MLIRLinalg
  MLIRLinalgToLLVM
  MLIRIR
  MLIRLLVMIR
  MLIRLoopOps
  MLIRLoopToStandard
  MLIRParser
  MLIRPass
  MLIRQuantOps
  MLIRSDBM
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRSupport
  MLIRTargetLLVMIR
  MLIRTargetLLVMIRModuleTranslation
  MLIRTransformUtils
  MLIRTransforms
  MLIRTranslateClParser
  MLIRTranslation
  MLIRVectorConversions
  MLIRVectorOps
  MLIRVectorToLoops
  MLIRVectorToLLVM
)

set(Lumen_libs
  ${Lumen_mlir_libs}
  ${Lumen_llvm_libs}
  lldCore
  lldCommon
  lldYaml
  lldReaderWriter
  lldELF
  lldWasm
  lldCOFF
  lldMachO
  lldMinGW
  lldDriver
)

add_llvm_library(Lumen
  ${globbed}

  DEPENDS
  OpsIncGen

  LINK_LIBS
  ${Lumen_libs}
)

llvm_update_compile_flags(Lumen)

whole_archive_link(Lumen ${Lumen_libs})

install(TARGETS Lumen
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
