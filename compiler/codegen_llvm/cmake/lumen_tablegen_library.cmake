include(CMakeParseArguments)

# lumen_tablegen_library()
#
# Runs lumen-tablegen to produce some artifacts.
function(lumen_tablegen_library)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME;TBLGEN"
    "TD_FILE;OUTS"
    ${ARGN}
  )

  if(NOT _RULE_TESTONLY OR LUMEN_BUILD_TESTS)
    # Prefix the library with the package name, so we get: lumen_package_name
    lumen_package_name(_PACKAGE_NAME)
    set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

    if(${_RULE_TBLGEN} MATCHES "LUMEN")
      set(_TBLGEN "LUMEN")
    else()
      set(_TBLGEN "MLIR")
    endif()

    set(LLVM_TARGET_DEFINITIONS ${_RULE_TD_FILE})
    set(_INCLUDE_DIRS ${LUMEN_COMMON_INCLUDE_DIRS})
    list(APPEND _INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
    list(TRANSFORM _INCLUDE_DIRS PREPEND "-I")
    set(_OUTPUTS)
    while(_RULE_OUTS)
      list(GET _RULE_OUTS 0 _COMMAND)
      list(REMOVE_AT _RULE_OUTS 0)
      list(GET _RULE_OUTS 0 _FILE)
      list(REMOVE_AT _RULE_OUTS 0)
      tablegen(${_TBLGEN} ${_FILE} ${_COMMAND} ${_INCLUDE_DIRS})
      list(APPEND _OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/${_FILE})
    endwhile()
    add_custom_target(${_NAME}_target DEPENDS ${_OUTPUTS})
    set_target_properties(${_NAME}_target PROPERTIES FOLDER "Tablegenning")

    add_library(${_NAME} INTERFACE)
    add_dependencies(${_NAME} ${_NAME}_target)

    # Alias the lumen_package_name library to lumen::package::name.
    lumen_package_ns(_PACKAGE_NS)
    add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})
  endif()
endfunction()
