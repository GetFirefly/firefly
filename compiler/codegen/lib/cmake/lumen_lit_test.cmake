include(CMakeParseArguments)

# lumen_lit_test()
#
# Creates a lit test for the specified source file.
#
# Parameters:
# NAME: Name of the target
# TEST_FILE: Test file to run with the lit runner.
# DATA: Additional data dependencies invoked by the test (e.g. binaries
#   called in the RUN line)
#
# A driver other than the default lumen/tools/run_lit.sh is not currently supported.
function(lumen_lit_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TEST_FILE"
    "DATA"
    ${ARGN}
  )
  if(NOT LUMEN_BUILD_TESTS)
    return()
  endif()

  lumen_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  get_filename_component(_TEST_FILE_PATH ${_RULE_TEST_FILE} ABSOLUTE)

  add_test(
    NAME ${_NAME}
    COMMAND ${CMAKE_SOURCE_DIR}/lumen/tools/run_lit.sh "${_TEST_FILE_PATH}"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" # Make sure the lit runner can find all the binaries
  )
  # TODO: Figure out how to indicate a dependency on _RULE_DATA being built
endfunction()


# lumen_lit_test_suite()
#
# Creates a suite of lit tests for a list of source files.
#
# Parameters:
# NAME: Name of the target
# SRCS: List of test files to run with the lit runner. Creates one test per source.
# DATA: Additional data dependencies invoked by the test (e.g. binaries
#   called in the RUN line)
#
# A driver other than the default lumen/tools/run_lit.sh is not currently supported.
function(lumen_lit_test_suite)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DATA"
    ${ARGN}
  )
  IF(NOT LUMEN_BUILD_TESTS)
    return()
  endif()

  foreach(_TEST_FILE ${_RULE_SRCS})
    get_filename_component(_TEST_BASENAME ${_TEST_FILE} NAME)
    lumen_lit_test(
      NAME
        "${_TEST_BASENAME}.test"
      TEST_FILE
        "${_TEST_FILE}"
      DATA
        "${_RULE_DATA}"
    )
  endforeach()
endfunction()
