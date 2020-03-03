lumen_cc_binary(
  NAME
    lumen_tblgen
  OUT
    lumen-tblgen
  SRCS
    "lumen-tblgen.cpp"
  DEPS
    lumen::compiler::Dialect::EIR::Tools
    MLIRSupport
    MLIRTableGen
    LLVMMLIRTableGen
    LLVMSupport
    LLVMTableGen
  LINKOPTS
    "-lpthread"
)
add_executable(lumen-tblgen ALIAS tools_lumen_tblgen)

if(${LUMEN_BUILD_COMPILER})

  # Additional libraries containing statically registered functions/flags, which
  # should always be linked in to binaries.
  #
  # TODO: Make the ALWAYSLINK property apply transitively and force
  #       MLIR libs to generally be ALWAYSLINK wherever used.
  set(_ALWAYSLINK_LIBS
    LLVMSupport
    MLIRAnalysis
    MLIREDSC
    MLIRParser
    MLIRPass
    MLIRStandardOps
    MLIRTransforms
    MLIRTranslation
    MLIRSupport
    lumen::compiler::Dialect::EIR::IR
  )

  foreach(LIB ${_ALWAYSLINK_LIBS})
    # Aliased targets are always in-project, so we control them and can set
    # ALWAYSLINK on them directly.
    # TODO: add the ALWAYSLINK property to MLIR libraries elsewhere
    get_target_property(_ALIASED_TARGET ${LIB} ALIASED_TARGET)
    if(NOT _ALIASED_TARGET)
      set_property(TARGET ${LIB} PROPERTY ALWAYSLINK 1)
    endif()
  endforeach(LIB)
endif()

if(${LUMEN_BUILD_TESTS})
  add_custom_target(LumenFileCheck ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/LumenFileCheck.sh LumenFileCheck
  )
  add_custom_target(LLVMFileCheck ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:FileCheck> FileCheck
    DEPENDS FileCheck
  )
endif()
